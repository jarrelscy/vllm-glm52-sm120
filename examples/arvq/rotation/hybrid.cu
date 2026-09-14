// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

// SM120 warp MMA: weights occupy the 16-row operand; eight independent
// activation groups (LUT mode) or token columns (direct mode) occupy N.
__device__ __forceinline__ void mma(float &d0,float &d1,float &d2,float &d3,
 unsigned a0,unsigned a1,unsigned a2,unsigned a3,unsigned b0,unsigned b1,
 unsigned sa=0x38383838u,unsigned sb=0x38383838u) {
 asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {0,0}, {%11}, {0,0};"
 : "+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3)
 : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),"r"(sa),"r"(sb));
}


__global__ void pack_planes(const half*in,unsigned*out,unsigned char*sc,int K,int slots,int P){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=slots*K)return;
 int k=i%K,slot=i/K;float v=__half2float(in[i]);
 for(int p=0;p<P;p++){
  float m=fabsf(v);for(int d=8;d;d/=2)m=fmaxf(m,__shfl_xor_sync(0xffffffff,m,d));
  int e=max(-6,min(8,(int)ceilf(log2f(fmaxf(m/6,0x1p-20f)))));float scale=exp2f((float)e),a=fabsf(v)/scale;
  unsigned q=(a>.25f)+(a>.75f)+(a>1.25f)+(a>1.75f)+(a>2.5f)+(a>3.5f)+(a>5.f);q|=(v<0)?8:0;
  const float tbl[8]={0,.5f,1,1.5f,2,3,4,6};float dec=tbl[q&7]*scale*((q&8)?-1.f:1.f);v=(v-dec)*16;
  unsigned bits=q<<(4*(k&7));for(int d=4;d;d/=2)bits|=__shfl_xor_sync(0xffffffff,bits,d);
  if((k&7)==0)out[((long long)slot*P+p)*(K/8)+k/8]=bits;
  if((k&15)==0)sc[((long long)slot*P+p)*(K/16)+k/16]=(e+7)<<3;
 }
}
extern "C" int hybrid_pack(const void*x,void*q,void*s,int K,int slots,int P,void*stream){pack_planes<<<(slots*K+255)/256,256,0,(cudaStream_t)stream>>>((const half*)x,(unsigned*)q,(unsigned char*)s,K,slots,P);return(int)cudaGetLastError();}
// Routes must be disjoint: cold_ids[slot]>=0 XOR hot_ids[slot]>=0.
// A slot with bothnegative is defined as zero; with bothpositive cold wins.
__global__ void hybrid_kernel(const unsigned* cw,const unsigned* cb,const unsigned char* cs,
 const unsigned* hw,const unsigned* hs,const unsigned* x,const unsigned* xs,
 const int* cold_ids,const int* hot_ids,float* partial,int N,int G,int S,int P){
 __shared__ unsigned lut[384];
 int slot=blockIdx.z,cold_e=cold_ids[slot],hot_e=hot_ids[slot];bool cold=cold_e>=0;
 if(cold){for(int i=threadIdx.x;i<384;i+=blockDim.x)lut[i]=cb[i];__syncthreads();}
 int lane=threadIdx.x&31,q=lane/4,c=lane%4,warp=threadIdx.x/32,tile=blockIdx.x*4+warp;
 if(tile>=N/16)return;
 float d0=0,d1=0,d2=0,d3=0;
 if(cold||hot_e>=0){
 for(int g=G*blockIdx.y/S;g<G*(blockIdx.y+1)/S;g++){
 unsigned b0=q<P?x[(((long long)slot*P+q)*G+g)*8+c]:0;
 unsigned b1=q<P?x[(((long long)slot*P+q)*G+g)*8+4+c]:0;
 unsigned sb=q<P?xs[((long long)slot*P+q)*G+g]:0x38383838u;
 unsigned a[4],r[4],sa;
 if(cold){
 const unsigned* w=cw+(((long long)cold_e*(N/16)+tile)*G+g)*60;
 #pragma unroll
 for(int j=0;j<4;j++){
 int bit=(j*32+lane)*15,word=bit/32,shift=bit%32;
 unsigned pair=__funnelshift_r(w[word],w[word+1],shift);
 a[j]=lut[pair&255];r[j]=lut[256+((pair>>8)&127)];
 }
 int row=q+8*(c&1);sa=cs[(((long long)cold_e*(N/16)+tile)*(G/2)+g/2)*16+row]*0x1010101u;
 mma(d0,d1,d2,d3,a[0],a[1],a[2],a[3],b0,b1,sa,sb);
 mma(d0,d1,d2,d3,r[0],r[1],r[2],r[3],b0,b1,sa,sb);
 }else{
 #pragma unroll
 for(int j=0;j<4;j++)a[j]=hw[(((((long long)hot_e*(N/16)+tile)*G+g)*4+j)*32)+lane];
 sa=hs[((long long)hot_e*N+tile*16+q+8*(c&1))*G+g];
 mma(d0,d1,d2,d3,a[0],a[1],a[2],a[3],b0,b1,sa,sb);
 }
 }
 }
 float a=ldexpf(d0,-8*c)+ldexpf(d1,-8*c-4),b=ldexpf(d2,-8*c)+ldexpf(d3,-8*c-4);
 a+=__shfl_xor_sync(0xffffffff,a,1);a+=__shfl_xor_sync(0xffffffff,a,2);
 b+=__shfl_xor_sync(0xffffffff,b,1);b+=__shfl_xor_sync(0xffffffff,b,2);
 if(c==0){partial[((long long)slot*N+tile*16+q)*S+blockIdx.y]=a;partial[((long long)slot*N+tile*16+q+8)*S+blockIdx.y]=b;}
}
__global__ void hybrid_reduce(const float* p,float* y,const int* cold_ids,const int* hot_ids,const float* hot_global,float cold_global,int N,int slots,int S,int hot_parts){
 long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(i>=(long long)N*slots)return;
 int slot=i/N,row=i%N;float v=0;for(int s=0;s<S;s++)v+=p[i*S+s];
 float scale=0;
 if(cold_ids[slot]>=0)scale=cold_global;
 else if(hot_ids[slot]>=0)scale=hot_global[(long long)hot_ids[slot]*hot_parts+row/(N/hot_parts)];
 y[i]=v*scale;
}
// C ABI (12pointers, float, 6ints, stream):
// cold_w:u32 [Ec,N/16,K/64,60]+1guard; cold_cb:u32[384];
// cold_scales:u8[Ec,N/16,K/128,16];
// hot_w:u32[Eh,N/16,K/64,4,32]; hot_scales:u32[Eh,N,K/64];
// hot_global:f32[Eh,hot_parts] (hot_parts=2 gate/up or1down);
// x:u32[slots,P,K/8]; xs:u8[slots,P,K/16] aliasu32[slots,P,K/64];
// cold_ids/hot_ids:i32[slots]; partial:f32[slots,N,split]; out:f32[slots,N].
// cold_global:f32; N,K,slots,split,P,hot_parts:int; CUDAstream:void*.
// N%16=0,K%128=0,P=1or4,hot_parts=1or2, split>0.
extern "C" int hybrid_launch(const void* cold_w,const void* cold_cb,const void* cold_scales,
 const void* hot_w,const void* hot_scales,const void* hot_global,const void* x,const void* xs,
 const void* cold_ids,const void* hot_ids,void* partial,void* out,float cold_global,
 int N,int K,int slots,int split,int P,int hot_parts,void* stream){
 if(N<=0||N%16||K<=0||K%128||slots<=0||split<=0||(P!=1&&P!=4)||(hot_parts!=1&&hot_parts!=2))return(int)cudaErrorInvalidValue;
 cudaStream_t s=(cudaStream_t)stream;
 hybrid_kernel<<<dim3((N+63)/64,split,slots),128,0,s>>>((const unsigned*)cold_w,(const unsigned*)cold_cb,(const unsigned char*)cold_scales,(const unsigned*)hot_w,(const unsigned*)hot_scales,(const unsigned*)x,(const unsigned*)xs,(const int*)cold_ids,(const int*)hot_ids,(float*)partial,N,K/64,split,P);
 cudaError_t err=cudaGetLastError();if(err!=cudaSuccess)return(int)err;
 hybrid_reduce<<<((long long)slots*N+255)/256,256,0,s>>>((const float*)partial,(float*)out,(const int*)cold_ids,(const int*)hot_ids,(const float*)hot_global,cold_global,N,slots,split,hot_parts);
 return(int)cudaGetLastError();}
