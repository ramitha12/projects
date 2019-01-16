# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 11:25:04 2016

@author: ramitha
"""
from __future__ import print_function
import supportfunc as st
import mojette 
import numpy as np
import scipy as sp
import scipy.signal as signal
from scipy import stats
import matplotlib.pyplot as plt
import imageio
import matplotlib.pyplot as plt
from matplotlib import colors
import numbertheory
import math

class Mlem(object):
    def __init__(self, projector, backprojector, g_j,epsilon,MRegPSF,angles,backFilt=False):
        """
        projector : function
            Function that takes an image and transforms it to
            projection space.
        backprojector : function
            Function that takes a sinogram and transforms it to
            image space.
        g_j : observed sinogram data.
        angles : ndarray of angles (in degrees) for projector and backprojector.

        """
        maxAngle = np.abs( np.array(angles) ).max() #max angle length
        print ("max angle:", maxAngle)
        self.correctRadius = int(abs(maxAngle)+0.5)
        self.backFilt=backFilt
        self.MRegPSF = int(MRegPSF)
        self.project = projector
        self.backproject = backprojector
        self.angles = angles
        self.g_j = g_j
        self.epsilon=epsilon
        
#        self.epsilon = np.mean(np.abs(g_j)) / 1000.0
#        self.sum_g_j = g_j.sum()
        self.i = 0
#        self.weighting = backprojector(g_j,256,angles)
        am=mojette.psf(angles,30,30,False,MRegPSF)
#        self.weighting=lut
        self.regpsf =  mojette.RegPSF(am,MRegPSF,epsilon,self.correctRadius)
        drt=self.regpsf
        drt[128,128]=0.017
        self.weighting=drt
        print ('regpsf',np.sum(self.regpsf))
        self.f =  (backprojector(g_j,self.MRegPSF,angles,self.regpsf,False))
#        self.g=np.zeros_like(self.f)
        self.g=self.f
        self.corr=self.weighting
#        assert self.f.shape == self.weighting.shape
        self.weighting.clip(self.epsilon, out=self.weighting)
        self.initF =   []
        self.initW =  self.weighting
        self.initsum=0
    def iterate(self):
        # From Lalush and Wernick;
        # f^\hat <- (f^\hat / |\sum h|) * \sum h * (g_j / g)          ... (*)
        # where g = \sum (h f^\hat)                                   ... (**)
        #
        # self.f is the current estimate f^\hat
        # The following g from (**) is equivalent to g = \sum (h f^\hat)
#        if self.i == 0:
#            self.initF=self.f
#            self.initW=self.weighting
#            print ('initial F',self.initF)
#            print ('initial W',self.initW)
#        if self.i==0:
#            self.initf=self.f
#        if self.i==0:
#           self.initF=self.f
#        self.f.clip(min=self.epsilon, out=self.f)
        ta=self.f
        g = self.project(self.f, angles)
        if self.i==0:
          self.initW=self.f
#        print ('mlem angles',angles)
#        for x in range(len(angles)):
#            print ('f len ','vect'+str(x) +' '+ str(len(self.g_j[x])))
#            print ('g len ','vect'+str(x) +' '+ str(len(g[x])))
#        g.clip(min=self.epsilon, out=g) important
        
        # form parenthesised term (g_j / g) from (*)
        
#        xg,yg=np.shape(g)
#        print ('shape xg', xg)
#        print ('shape yg', yg)
        rlist=[]
        for x in range(len(angles)):
#           print('g[x]',g[x])
           
#           g[x].clip(min=(self.epsilon),max=(g_j[x].max()),out=g[x])
           hyp=g[x]
#           hyp = hyp - np.abs(hyp.min())
           lk=g_j[x]
           kl=(hyp/hyp.max())*(lk.max())+ lk.min()
#           g[x].clip(min=(self.epsilon), out=g[x])
           print('hyp max',hyp.max())
           print('hyp min',hyp.min())
           print('g[x] max',kl.max())
           print('g[x] min',kl.min())
           print('lk[x] max',lk.max())
           print('lk[x] min',lk.min())
           r = np.divide(self.g_j[x],kl)  
           rlist.append(r)
        self.g=rlist
#        print('g',self.g)
        # backproject to form \sum h * (g_j / g)
        g_r = self.backproject(rlist,int(self.MRegPSF),self.angles,self.regpsf,self.backFilt)
        
        # Renormalise backprojected term / \sum h)
        # Normalise the individual pixels in the reconstruction
#        self.corr *=self.weighting       
        tm= (g_r) / (self.weighting)
        ab= self.f * tm
        self.f=ab
        self.initF.append(self.f)
        print ('diff of image', np.sum(ta - self.f))
        print ('diff of image', np.sum(g_r))
#        self.initF.append(self.f)
        # print some progress output
        if 0:
            print('.', end='')
        else:
            self.stats()

        self.i += 1
        return self.f

    def _imshow(self, im, show=True):
        plt.figure()
        plt.imshow(im, interpolation='nearest', cmap='YlGnBu_r')
        plt.title('iteration:%d' % self.i)
        plt.colorbar()
        if show:
            plt.show()

    def stats(self):
        print('i={i} f_sum={f_sum} f_min={f_min} f_max={f_max} '
              .format(
                  i = self.i,
                  f_sum = self.f.sum(),
                  f_min = self.f.min(),
                  f_max = self.f.max(),
#                  g_j_sum = self.sum_g_j,
                  )
              )

    def imsave_f(self):
        imageio.imsave('mlem_mpl_%03d.tif' % self.i, self.f.astype(np.float32))
#    def calculateEpsilon(self,invect):
#        vect_mean=np.zeros(len(invect),dtype=np.float64)
#        for x in range(len(invect)):
#            vect_mean[x]=np.mean(np.abs(invect[x]))
##            print ('mean vect',vect_mean[x])
#        mean_proj = np.max(vect_mean)
#        print ('mean', mean_proj)
#        return mena_proj
#    def normaX(self,image):
#        x=sp.norma(image)
#        return x
def mse(refImage,OutImage):
    x,y=np.shape(OutImage)
    tot_mse=0
    for i, row in enumerate(OutImage):
       for j, col in enumerate(row):
           tot_mse= np.square(refImage[i,j]- OutImage[i,j])+tot_mse         
    return tot_mse/(x*y)       
    
def psnr(refImage,OutImage,maxPixel=255):
    mse_out=mse(refImage,OutImage)
    psnr_out=20 * math.log(maxPixel / math.sqrt(mse_out), 10)
    return psnr_out

def addAWGN_projs(Inprojs,SNR_DB):
    out=[]
    for x in range(len(Inprojs)):
        proj = np.asarray(Inprojs[x])
        out_proj=addAWGN(proj,SNR_DB)
        out.append(out_proj)
    return out
        
def addAWGN(Insignal,SNR_DB):
    desired_SNR=10**(SNR_DB/10)
    Insignal_len= len(Insignal)
    awgnNoise = np.random.rand(Insignal_len)
    Sig_pwr = np.sqrt(np.sum(np.square(Insignal)))/Insignal_len
    Noise_pwr = np.sqrt(np.sum(np.square(awgnNoise)))/Insignal_len
    if desired_SNR!= 0:
       scalemult = (Sig_pwr/Noise_pwr)/desired_SNR
       awgnNoise = awgnNoise*scalemult
       out_signal = Insignal + awgnNoise
#       print ('out',out_signal)
#       print ('Insignal',Insignal)
#       print ('awgnNoise',awgnNoise)
##       print ('sig_max',sig_max)
    else:
       out_signal = awgnNoise
    return out_signal
    
def noisify(projectionData, frac=0.1):
    # Add Poisson detector noise; Clip just above 0 to avoid scipy bug:
    # See https://github.com/scipy/scipy/issues/1923
    projectionData.clip(projectionData.max()/1e6, out=projectionData)
    I_MAX = projectionData.max()
    sigma = frac * sp.ndimage.standard_deviation(projectionData) / I_MAX
    # rescale for noise addition, create noisy version and rescale
    projectionData = projectionData / (I_MAX * (sigma**2))
    projectionData = stats.poisson.rvs(projectionData) * I_MAX * (sigma**2)
    return projectionData
def backproj_norm(backProj,maxVal=205):
    return(st.norma(backProj-backProj.min())*maxVal)    
def ImageStats(outImage,refImage,img_size,maxPixel=255,maxRefVal=205):
    x,y=np.shape(refImage)
    stpt=int(x/2)-int(img_size/2)
    endpt=int(x/2)+int(img_size/2)
    outImg_norm=backproj_norm(outImage,maxRefVal)
    outImg_norm=outImg_norm[stpt:endpt,stpt:endpt]
    refImage=refImage[stpt:endpt,stpt:endpt]
    diff_img = outImg_norm - refImage
    mse_out = mse(refImage,outImg_norm)
    psnr_out = psnr(refImage,outImg_norm)
    return diff_img,mse_out,psnr_out
    
    
def generategraph(test_name,parameter_val,text_parameters,PSNR_list=[],MSE_list=[],diff_list=[],filt_list=[],noise_img=[],f0=[]):
    plt.figure(1)
    plt.subplot(121)
    plt.imshow(f0, interpolation='nearest', cmap='gray')
    plt.title('original Image') 
    plt.subplot(122)
    plt.imshow(noise_img , interpolation='nearest', cmap='gray')
    plt.title('backProj Image') 
    parameter_len=len(parameter_val)
    array_len=len(MSE_list[0])
    iter_array=np.arange(0, array_len,1)
    print('iter_array',iter_array)
    for p in range(parameter_len):
        MSE_array=MSE_list[p]
        plt.figure(2)
        plt.plot(iter_array,MSE_array)    
    plt.ylabel('MSE')
    plt.xlabel('iterations')
    plt.title(text_parameters)     
    
    
if __name__ == '__main__':

    import skimage
    from skimage.transform import radon, iradon, downscale_local_mean
    from skimage.util import pad
    from skimage import data, filters

    '''
    # Make a phantom that is 0 everywhere except for a small 3x3 block
    f0 = np.zeros((32, 32))
    f0[16:19, 16:19] = 1
    '''

    # '''
    # lena
    N = 30
    M = 64
    f0 = np.zeros((256, 256))
    f0 = imageio.lena(N,256)

    def angle_set(katz,N,padN,angleset_type=1,quad=2,octant=2):
    
        angles=[]
        if  (angleset_type==1):
            if katz >0 :
                prime_num = int(numbertheory.nearestPrime(katz*N))
                print ("prime number:", prime_num)
                angles = mojette.angleSet_Finite(prime_num,quad)
        if angleset_type==2 :  
            angles, lengths = mojette.angleSet_Symmetric(N,N,octant,True,katz)
        if angleset_type==3 :  
            angles=angleSet_MinimalL1(N, N, octant,False,katz)
        return angles 
        
        
    def projector(x, angles):
        '''
        skimage's radon transform

        '''

        print('rite project')
        y=mojette.transform(x, angles)
        return y


    def backprojector(x,MRegPSF,angles,RegPSFh=[],backFilt = False):
        '''
        skimage's fbp-based inverse radon transform

        '''
        print('rite backproject')
        # On the next line, the filter=None selection is *very* important
        # as using the ramp filter causes the scheme to diverge!
        hy=mojette.backprojectPadded(x, angles, MRegPSF,MRegPSF,MRegPSF,MRegPSF, True, True)
#        scale=hy.max()
        if backFilt== True:
           ab = mojette.deconvolve(st.norma(hy),RegPSFh)
#           ab.clip(min=0, out=ab)
           jk= ab + np.abs(ab.min())
#           y=st.norma(ab)*(205)
           y=st.norma(jk)*(hy.max())+hy.min()
           print ('backproj ',np.sum(hy))
           print ('filt deconvo',np.sum(ab))
           print('deconv max',ab.max())
           print('deconv max',ab.min())
           print ('deconvo normalised',np.sum(y))
           print ('backproj ',hy.max())
           print ('backproj ',hy.min())
           print ('norma min',y.min())
           print ('norma max',y.max())
#           print ('deocn')
        else:
           y = hy
        return y



##########################################my stuff#######################################################  



    katz=0.8
    N=30
    M=64
    Npad=4*M
    octant=2
    quad=2
    angleset_type=2
    SNR_DB=5
    backFilt=True 
    parameter_val =[5]
    
    angles = angle_set(katz,N,Npad,angleset_type,quad,octant)
#    backFilt = True
#    MRegPSF=4*M
#    k=1
#    p = numbertheory.nearestPrime(k*N)
#    print ("p:", p)
#    angles = mojette.angleSet_Finite(p, 2)
#    mu = len(angles)
#    maxAngle = np.abs( np.array(angles) ).max()
#    print ("max angle:", maxAngle)
#    correctRadius = int(abs(maxAngle)+0.5) #round up
    
    
    
    PSNR_list=[]
    MSE_list=[]
    diff_list=[]
    filt_list=[]
    num_iter=10
    SNR_DB=5
    parameter_val =[10]
    parameter_len=len(parameter_val)
    epsilon =0.01587302
    g_j = projector(f0,angles)
    g_j_back=backprojector(g_j,Npad,angles,[],False)
#    for x in range(parameter_len):
    g_j_noise = addAWGN_projs(g_j,parameter_val[0])
    backproj_noise=backprojector(g_j_noise,Npad,angles,[],False)  
    mlem = Mlem(projector, backprojector, g_j_noise,epsilon,Npad,angles,True)
    diff_image=[]
    filt_image=[]
    PSNR=np.zeros((num_iter), dtype=np.float)
    MSE=np.zeros((num_iter), dtype=np.float)
    hy=[]
    for im in range(num_iter):
        hy.append(mlem.iterate())
        gy=hy[im]
        diffimage,MSE[im],PSNR[im] = ImageStats(gy,f0,N,255,205)
        diff_image.append(diffimage)
        filt_image.append(gy)
    PSNR_list.append(PSNR) 
    MSE_list.append(MSE)
    diff_list.append(diff_image)
    filt_list.append(filt_image)  
    
text_parameters='katz'+str(katz) +'  ' + 'symmetric'+'  '   
generategraph('noise',parameter_val,text_parameters,PSNR_list,MSE_list,diff_list,filt_list,backproj_noise,f0)

##############################################################################################
cmap = colors.ListedColormap(['black', '0.34', 'white'])
mu=len(angles)
bounds=[-1.0/(mu-1.0),-1e-3,1e-3,1.0]
norm = colors.BoundaryNorm(bounds, cmap.N)  
plt.figure(1)
plt.subplot(121)
plt.imshow(f0, interpolation='nearest', cmap='gray')
plt.title('original Image') 
plt.subplot(122)
plt.imshow(g_j_back, interpolation='nearest', cmap='gray')
plt.title('backProj Image') 
plt.figure(2)
plt.imshow(backproj_noise, interpolation='nearest', cmap='gray')
plt.title('additive white gaussian noise per projection snr '+str(SNR_DB)+'DB ' + '     noise backProj Image' + '  psnr='+str("%0.2f" %psnr_noise) + '  mse='+str("%0.2f" %mse_noise)) 
#plt.suptitle('additive white gaussian noise snr_DB'+str(SNR_DB))

num_iter_show=[0,2,4,6,9]
tot_fig=3
for x in range(len(num_iter_show)):
    itr=num_iter_show[x]
    plt.figure(tot_fig)
    num_fig= x%2
    tot_fig=tot_fig + num_fig
    sub_fig=121 + num_fig
    plt.subplot(sub_fig)
    plt.imshow((hy[itr])[113:143,113:143], interpolation='nearest', cmap='gray')
    plt.title('iteration'+str(itr)+'  psnr='+str("%0.2f" %psnr_out[itr]) + '  mse='+str("%0.2f" %mse_out[itr])) 
    plt.suptitle('reconstructed')
    
plt.figure(tot_fig)
plt.subplot(131)
plt.imshow(f0[113:143,113:143], interpolation='nearest', cmap='gray')
plt.title('original Image') 
plt.subplot(132)
plt.imshow(backproj_noise[113:143,113:143], interpolation='nearest', cmap='gray')
plt.title('noise backProj Image' + '  psnr='+str("%0.2f" %psnr_noise) + '  mse='+str("%0.2f" %mse_noise)) 
plt.subplot(133)
plt.imshow((hy[itr])[113:143,113:143], interpolation='nearest', cmap='gray')
plt.title('iteration'+str(itr)+'  psnr='+str("%0.2f" %psnr_out[itr]) + '  mse='+str("%0.2f" %mse_out[itr]))  
plt.suptitle('additive white gaussian noise per projection snr '+str(SNR_DB)+'DB ')

#plt.figure(2)
#plt.suptitle('reconstructed')
#itr=0
#plt.subplot(121)
#plt.imshow((hy[itr])[113:143,113:143], interpolation='nearest', cmap='gray')
#plt.title('iteration'+str(itr)+'  psnr='+str("%0.2f" %psnr_out[itr]) + '  mse='+str("%0.2f" %mse_out[itr])) 
#plt.suptitle('reconstructed')
#itr=2
#plt.subplot(122)
#plt.imshow((hy[itr])[113:143,113:143], interpolation='nearest', cmap='gray')
#plt.title('iteration'+str(itr)+'  psnr='+str("%0.2f" %psnr_out[itr]) + '  mse='+str("%0.2f" %mse_out[itr])) 
#plt.suptitle('reconstructed')
#itr=4
#plt.figure(3)
#plt.subplot(121)
#plt.imshow((hy[itr])[113:143,113:143], interpolation='nearest', cmap='gray')
#plt.title('iteration'+str(itr)+'  psnr='+str("%0.2f" %psnr_out[itr]) + '  mse='+str("%0.2f" %mse_out[itr])) 
#plt.suptitle('reconstructed')
#itr=6
#plt.subplot(122)
#plt.imshow((hy[itr])[113:143,113:143], interpolation='nearest', cmap='gray')
#plt.title('iteration'+str(itr)+'  psnr='+str("%0.2f" %psnr_out[itr]) + '  mse='+str("%0.2f" %mse_out[itr]))
#plt.figure(4)
#plt.suptitle('reconstructed') 
#itr=9
#plt.subplot(121)
#plt.imshow((hy[itr])[113:143,113:143], interpolation='nearest', cmap='gray')
#plt.title('iteration'+str(itr)+'  psnr='+str("%0.2f" %psnr_out[itr]) + '  mse='+str("%0.2f" %mse_out[itr])) 
#plt.subplot(122)
#plt.imshow(f0[113:143,113:143], interpolation='nearest', cmap='gray')
#plt.title('original Image') 
#xaxis=np.arange(0, num_iter,1)
#psnryaxis=np.zeros(num_iter)
#mseyaxis=np.zeros(num_iter)
#for x in range(num_iter):
#    psnryaxis[x]=psnr_out[x]
#    mseyaxis[x]=mse_out[x]
#plt.figure(tot_fig+1)
#plt.subplot(121)
#plt.plot(xaxis,psnryaxis)  
#plt.title('psnr vs iteration')
#plt.subplot(122)
#plt.plot(xaxis,mseyaxis)  
#plt.title('mse vs iteration')




##################################################################################
#plt.subplot(122)
#plt.imshow(hy[5], interpolation='nearest', cmap='gray')
#plt.title('iteration 1') 
#plt.figure(3)
#plt.subplot(121)
#plt.imshow(np.abs(st.fft2d(st.norma(f0.astype('float')))), interpolation='nearest', cmap='gray')
#plt.title('iteration 2') 
#plt.subplot(122)
#plt.imshow(np.abs(st.fft2d(st.norma(hy[3]))), interpolation='nearest', cmap='gray')
#plt.title('iteration 3') 
#plt.figure(4)
#plt.imshow(diff_image, interpolation='nearest', cmap='gray')
#plt.title('image difference') 
#from mpl_toolkits.mplot3d import Axes3D
#fig = plt.figure()
#ax  = fig.gca(projection='3d')
##
#X = np.arange(0, MRegPSF)
#Y = np.arange(0, MRegPSF)
#X, Y = np.meshgrid(X, Y)
#surf = ax.plot_surface(X, Y,diff[3], rstride=1, cstride=1, cmap='jet',
#        linewidth=0, antialiased=True)
#plt.figure(2)
#plt.subplot(121)
#plt.imshow(mlem.initF, interpolation='nearest', cmap='gray')
#plt.title('initial F') 
#plt.figure(3)
#plt.imshow(mlem.g_j, interpolation='nearest', cmap='gray')
#plt.title('initial W')   