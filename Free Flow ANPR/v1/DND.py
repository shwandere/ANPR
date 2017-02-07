import cv2
import numpy as np
import os
import sys
import glob
import time
from scipy import stats
from array import array
from sklearn.decomposition import RandomizedPCA
from sklearn.externals import joblib
import string
from perspective_transform import perspective_transform
from haar_classifier import haar_classifier
from naomi_classify import deep_classify
comp1=0
global path
global f
svm_calling=0
class StatModel(object):
    '''parent class - starting point to add abstraction'''    
    def load(self, fn):
        self.model.load(fn)
    def save(self, fn):
        self.model.save(fn)

class SVM(StatModel):
    '''wrapper for OpenCV SimpleVectorMachine algorithm'''
    def __init__(self):
        self.model = cv2.SVM()

    def train(self, samples, responses):
        #setting algorithm parameters
        params = dict( kernel_type = cv2.SVM_RBF, 
                       svm_type = cv2.SVM_C_SVC,
                       C=2.67,
                       gamma=5.383)
        self.model.train(samples, responses, params = params)

    def predict(self, samples):
        #return np.float32( [self.model.predict(s) for s in samples])
        return self.model.predict(samples)

def find_lh(a,b,bool1):

    if a>b:
        if bool1==1:        return a
        else:   return b
    else:
        if bool1==0:        return a
        else:   return b


def plate_color(img):
    imgarea = img.size
    #r,g,b = cv2.split(img)
    l=[]*8
    l.append([[236,163,169],[255,255,195]])
    l.append([[172,127,62],[181,145,146]])
    l.append([[169,182,115],[255,245,170]])
    l.append([[151,81,92],[255,255,135]])
    l.append([[151,111,92],[181,133,127]])
    l.append([[212,144,4],[251,180,147]])
    l.append([[126,106,0],[250,216,76]])
    l.append([[125,87,41],[160,121,135]])
    l =np.array(l)
    
    for i in range(8):
        #print l[i][0]
        #print l[i][1]
        product = cv2.inRange(img,l[i][0],l[i][1])
        #print img.dtype
        '''
        temp = np.multiply(r,g)
        product = np.multiply(temp,b)
        '''
        if (np.sum(product)/255)/img.size > .2:
            return 1
        else:
            return 0
def sauvola_threshold(img):
    
    i=0
    j=0

    r_size=20
    rows=img.shape[0]
    cols=img.shape[1]
    mean=np.zeros(shape=[rows/r_size+1,cols/r_size+1])
    std=np.zeros(shape=[rows/r_size+1,cols/r_size+1])
    t=np.zeros(shape=[rows/r_size+1,cols/r_size+1])

    img1=np.zeros((rows,cols),np.uint8)

    for x in range(0,rows,r_size):
        for y in range(0,cols,r_size):
            ##print x,y
            if x>rows or y>cols:
                continue
            if x+r_size>rows or y+r_size>cols:
                sliced_img=img[x:rows,y:cols]
            else:
                sliced_img=img[x:x+r_size,y:y+r_size]
            mean[i,j]=np.mean(sliced_img)
            std[i,j]=np.std(sliced_img)
            #t[i,j]=0.5*mean[i,j]+0.5*img.min()+0.5*((std[i,j]/img.max())*(mean[i,j]-img.min()))
            j=j+1
        j=0
        i=i+1
    i=0
    j=0
    k=0.5
    #####sauvola######
    
    R=std.max()
    M=img.min()
    t1=np.multiply((1-k),mean)
    t2=k*M
    t3=np.multiply(k,np.multiply((np.divide(std,R)),(np.subtract(mean,M))))
    t=np.add(t1,np.add(t2,t3))
    return t

def cal_height(chars,char_x,char_w,char_h):

    count=0
    sum1=0
    mean=0
    
    if len(char_h)==0:
        return None

    try:
        #comp=stats.mode(char_h).mode[0]                
        comp=np.floor(np.mean(char_h))
        #print char_h
        #print comp
        #print 'comp'
        
    except:                     
        comp=char_h[0]
    chars1=[]
    x=[]
    comp_list=[comp]
    for i in range(len(chars)):
        cv2.imshow("final",chars[i])
        #print char_h[i]
        #cv2.waitKey(0)
    for i in range(len(char_h)):
        if np.absolute(comp-char_h[i])<=10:
            if np.absolute(comp-min(comp_list))<=15:
                    comp_list.append(char_h[i])
                    #if char_h[i]>comp:
        '''          #   comp=char_h[i]
        else:
            bigger.append(i)
            count=count+1
        
        '''
        #print comp_list
        comp=np.mean(comp_list)
    temp=[]
    h=[]
    for i in range(len(char_h)):
        if (np.absolute(comp-char_h[i]))>=10 and char_h[i]<comp :
            continue
        chars1.append(chars[i])
        x.append(char_x[i])
        h.append(char_h[i])
        temp.append(char_w[i]/float(char_h[i]))
    aspect_ratio=np.array(temp) ##TO SORT THE ASPECT RATION ARRAY USING INDEX OF SORTED X  
    #print len(chars1)                  
    
    #return chars1
    
    if len(x)>3:
        
        dict_x=dict(zip(range(0,len(x)),x))
        
        temp_x=sorted(dict_x,key=dict_x.__getitem__)
        len1=len(temp_x)
        #print temp_x
        
        aspect_ratio=aspect_ratio[np.array(temp_x)]
        h=np.array(h)
        h=h[temp_x]
        ldel=[]
        #print aspect_ratio
        for i in range(len(aspect_ratio)):
            #print comp1
            if ((i+1)!=len(aspect_ratio) and (np.absolute(aspect_ratio[i]-aspect_ratio[i+1])>.3) ) or np.absolute(aspect_ratio[i]-comp1)>.25:
                #or np.absolute(h[i]-h[i+1])>15
                #print "yes"
                #print aspect_ratio[i]
                ldel.append(temp_x[i])####np.delete to remove array elements takes two arguments array and index
            
            else:break
        #print ldel  
        for i in range(len(aspect_ratio)):
            if ((len(aspect_ratio)-i-1)!=0 and np.absolute(aspect_ratio[len(aspect_ratio)-i-1]-aspect_ratio[len(aspect_ratio)-i-2])>.3) or np.absolute(aspect_ratio[len(aspect_ratio)-i-1]-comp1)>.25:
                #print "yes"
                #or np.absolute(h[len(aspect_ratio)-i-1]-comp)>15
                #print aspect_ratio[len(aspect_ratio)-i-1]
                ldel.append(temp_x[len(aspect_ratio)-i-1])  
            else: break
        #print len(chars1)
        #print ldel          
            #np.delete(aspect_ratio,len(aspect_ratio)-i)
         
        #####sort the aspect_ratio according to x
        
        sorted_chars=[]
        j=0
        h1=[]
        for i in xrange(len(chars1)-1,-1,-1):
            if temp_x[i] not in ldel:
                sorted_chars.append(chars1[temp_x[i]])
                h1.append(h[i])
            j=j+1
        
        temp_x=[]
        temp_x=x1
        ind_1=0
        ind_2=1
        ind_l=len(sorted_chars)-1
        ind_lminus1=len(sorted_chars)-2
        #print 'x.shape,len(final_chars)'
        #print len(x),len(final_chars)
        #print"[ind_1],[ind_2],[ind_l],[ind_lminus1]"
        ##print h1[ind_1],h1[ind_2],h1[ind_l],h1[ind_lminus1]
        #print h1
        if len(sorted_chars)>4:
            #print abs(h1[ind_1]-h1[ind_2])
            #print len(sorted_chars)
            if abs(h1[ind_1]-h1[ind_2])>10:
                #print "yes_ height_difference"# cannot be smaller than this as bolted chars can be present
                del sorted_chars[ind_1]
                del h1[ind_1]
                ind_l=ind_l-1
                ind_lminus1=ind_lminus1-1
                ##print h1[ind_l],h1[ind_lminus1]
            if abs(h1[ind_l]-h1[ind_lminus1])>10:
                #print "yes_height_difference"
                del sorted_chars[ind_l]
                del h1[ind_l]
        
        return sorted_chars
    else:
        return None
    

def cal_asp(w,h,svm_label):
    global comp1
    global svm_calling
    count=0
    sum1=0
    mean=0
    
    aspect_ratio=[]
    
    for j in range(0,len(w)):
        if svm_label[j]==1 or svm_label[j]==2:
            aspect_ratio.append(w[j]/float(h[j]))
    if len(aspect_ratio)==0:
        return None
    bigger=[]
    #aspect_ratio=np.divide(w,float(h[j])
    #print "this is the aspect ratio"
    #print aspect_ratio
    try:
        #comp=stats.mode(aspect_ratio).mode[0]
        comp=np.mean(aspect_ratio)
    except:
        comp=aspect_ratio[0]
    #comp=min(aspect_ratio)
    
    comp_list=[comp]
    #print comp,min(comp_list)
    for i in range(len(aspect_ratio)):
        
        if np.absolute(comp-aspect_ratio[i])<.2:
            if np.absolute(comp-min(comp_list))<.4:
                    comp_list.append(aspect_ratio[i])
                    #if aspect_ratio[i]>comp:
                    #   comp=aspect_ratio[i]
            else:
                if comp-min(comp_list)>0:
                    bigger.append(aspect_ratio[i])
                    count=count+1
        else:
            bigger.append(aspect_ratio[i])
            count=count+1
        comp=np.mean(comp_list)
        
    
    
    #print "bigger"+str(bigger)
    #print "comp_list"+str(comp_list)

    ##################################################small chars
    for j in range(len(bigger)):
        if bigger[j]<comp:
            #print j
            ind=aspect_ratio.index(bigger[j])
            #print x
            #print x[ind]
            #if x[ind]!=min(x):
            comp_list.append(bigger[j])
            
            #del bigger[j]
    comp=np.mean(comp_list)
    #print comp_list
    if svm_calling==0:
        #print "yes_svm"
        comp1=comp
    #print "svm_calling"
    #print svm_calling
    #print "comp1"
    #print comp1
    return comp_list
    
pca=RandomizedPCA(n_components=1000)
svm1=cv2.SVM()
svm1.load("char_seg3_9 orientation_16x16 cell size_new_chars1.dat")
svm2=SVM()
svm2.load("char_seg3_9 orientation_16x16 cell size_new_chars5_learn_svm_kernel.dat")
hog=cv2.HOGDescriptor((64,64),(64,64),(16,16),(16,16),9)
pca=joblib.load('filename.pkl')


def svm_calc(given_ch,img_svm):
    char_cont,hierarchy=cv2.findContours(img_svm.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    ##print char_cont
    h=[0]*len(char_cont)
    x=[0]*len(char_cont)
    y=[0]*len(char_cont)
    w=[0]*len(char_cont)
    svm_label=[0]*len(char_cont)
    chars=[0]*len(char_cont)
    count=0
    s1=0
    ##print chars
    
    for i in range(len(char_cont)):
        
        x[i],y[i],w[i],h[i]=cv2.boundingRect(char_cont[i])
        ##print char_cont[i]
        sub=np.zeros(img_svm.shape,np.uint8)
        cv2.drawContours(sub,char_cont,i,(255,255,255),-1)
        temp=sub[y[i]:y[i]+h[i],x[i]:x[i]+w[i]]
        new=img_svm[y[i]:y[i]+h[i],x[i]:x[i]+w[i]]
        sub3=cv2.bitwise_and(new,temp)
        ret,sub3=cv2.threshold(sub3,127,255,cv2.THRESH_BINARY)
        s2=0
        if given_ch!=None:
            chars[count]=given_ch[y[i]:y[i]+h[i],x[i]:x[i]+w[i]]
        else:
            chars[count]=sub3
        name=path+"/"+"output"+"/"+f[len(path)+1:len(f)-4]+"/"+"char_"+str(s2)+".jpg" #f[len(path)+1:len(f)-4]+
        s2=s2+1
        #cv2.imwrite(name,sub.copy())
        ##cv2.imshow("sub",sub3)
        
      ##  cv2.waitKey(0)
        count=count+1
        s1=s1+1
    
    
    for i in range(len(char_cont)):
        
        
        svm_op=[]
        rows=chars[i].shape[0]
        cols=chars[i].shape[1]
        if rows>size or cols>size:
            continue
        img0=cv2.copyMakeBorder(chars[i],size-rows/2,size-rows/2,size-cols/2,size-cols/2,cv2.BORDER_CONSTANT,value=[0,0,0])
        des=(hog.compute(img0))#,(1,1),(0,0)))
        
        #t=np.empty((1,9216),np.float32)
        #t[0][:]=des[:,0]
        #t=[]
        #t[0]=des[:,0]
        
        #pca.fit(np.transpose(des))
        svm_op.append(svm1.predict(des))
        svm_op.append(svm2.predict(pca.transform(np.transpose(des))))
        
        
        if svm_op[0]!=svm_op[1]:
            svm_label[i]=max(svm_op)
        else:
            svm_label[i]=svm_op[0]
        del svm_op[:]



        '''
        ##############noise removaL
        area1=cv2.contourArea(char_cont[i])
        
        #print "area"+str(area1)
        _,_,width,height=cv2.boundingRect(char_cont[i])
        bounding_area=width*height
        convex_hull=cv2.convexHull(char_cont[i])
        convex_area=cv2.contourArea(convex_hull)
        #print "convex_area"+str(convex_area)
        #print "bounding_area"+str(bounding_area)
        extent=area1/float(bounding_area)

        m02=cv2.moments(char_cont[i])['mu02']
        m20=cv2.moments(char_cont[i])['mu20']
        m11=cv2.moments(char_cont[i])['mu11']
        #bigSqrt = np.sqrt( ( m20 - m02 ) *  ( m20 - m02 )  + 4 * m11 * m11  )
        #eccentricity = ( m20 + m02 + bigSqrt ) /float ( m20 + m02 - bigSqrt )
        
        
        if (extent<.2 or extent>0.9):
            ##print "solidity is less than 100"
            svm_label[i]==-1
        if convex_area!=0:
            solidity=area1/float(convex_area)
            if solidity<.3:
                
                svm_label[i]==-1
        #print len(char_cont[i])
        if len(char_cont[i])>5:
            
            _,axes,_= cv2.fitEllipse(char_cont[i])
            mx=min(axes)
            MX=max(axes)
            eccentricity = np.sqrt(1-(mx/float(MX))**2)
            if eccentricity >0.9 or eccentricity <0.8:
                svm_label==-1
            cv2.imshow('char_cont',img_svm[y[i]:y[i]+h[i],x[i]:x[i]+w[i]])
            #print eccentricity
            #cv2.waitKey(0)
        '''
        
    x1=np.array(x)   
    y1=np.array(y)
    h1=np.array(h)
    w1=np.array(w)
    s1=np.array(svm_label)
    if h and (len(s1[s1==1]+len(s1[s1==2])))!=0:
        temp_y=y1[s1==1]
        temp_y=np.concatenate((temp_y,y1[s1==2]),axis=0)
        #print temp_y.shape
        temp_h=h1[s1==1]
        temp_h=np.concatenate((temp_h,h1[s1==2]),axis=0)
        temp_w=w1[s1==1]
        temp_w=np.concatenate((temp_w,w1[s1==2]),axis=0)
        s2=0
        #min_y=min(temp_y)
        min_y=np.mean(temp_y)
        #max_y=max(np.add(temp_y,temp_h))
        max_y=np.mean(np.add(temp_y,temp_h))
        #print min_y,max_y
        
    x2=x1[s1==1]
    x2=np.concatenate((x2,x1[s1==2]),axis=0)        
    cal_aspr=cal_asp(w,h,svm_label)
    if not h:
        chars=None
        svm_label=None
        cal_aspr=None  
    return chars,svm_label,cal_aspr,w,x,y,h




###############################################################    


size=90 
s2=0
max_x=0
min_x=0
cndtn=False
i=0
j=0
min_indx=0
min_indy=0
p=[0]                                                                                                                                                                     
q=[0]

input_path = open('input.txt','r')
f1 = input_path.readline() 
print f1
path = f1[0:len(f1)-f1[len(f1):0:-1].index('\\')]
print path
output = path+'/'+'output'+'/'
if os.path.isdir(output)==False:
    os.mkdir(output,0777)
#path="C:/Users/R&D/Desktop/"
#path="F:/New folder3"
#file1=open("D:/lps/output/errorneous/missing/filenames.txt")
#path= "D:/lps"#output_best/errorneous/chars missing/img"
#path=raw_input("give vfolder path for lps")

output=path+"/"+"output"+"/"                                                                                                                                                                                                                                                 
if os.path.isdir(output)==False:
        os.mkdir(output,0777)
#D:/lp rois/lp7/output/errorneous/chars missing/missing
#C:/Users/R&D/Desktop              
##print  2  
#files=glob.glob(os.path.join(path,"*.jpg"))


#file1=open("D:/v71_lps/output/errorneous/need to be correct/filename.txt",'r')
'''
#file1.writelines(string.split(files))
for i in (files):
    #print i
    file1.write(str(i))
file1.close()
file2=open("D:/lps/10_files/filename.txt",'r')
'''
comp_count=0
s1=0
r_size=20
file1=open("output.txt",'w')
img0=np.zeros((64,64),np.uint8)
loop=0
f=f1
while loop==0:
    '''
    #print f
    f=f.rstrip("\n")
    f=f[48:len(f)]
    f=path+'/'+f+'.jpg'
    '''
    
    #loop=loop+1
   
    f=f1
    i=0
    j=0
    svm_calling=0
    
    print f
    #************perspective_transform and haar classifier to extract LP***********#
    roi=cv2.imread(f)

    
    #transformed = perspective_transform(roi)
    #extracted_haar_lp=haar_classifier(transformed)
    color = plate_color(roi) ######binary 1 for yellow and 0 for white
    img = roi

    img1=img.copy()
    rows=img.shape[0]
    cols=img.shape[1]
    vertical=np.zeros((rows,cols),np.uint8)
    sobelt=np.zeros((rows,cols),np.uint8)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    cv2.imshow('img.jpg',img)  
    ##cv2.waitKey(0)
    #img=cv2.equalizeHist(img)
    img = cv2.medianBlur(img,3)
    blur=cv2.GaussianBlur(img,(3,3),0,0,cv2.BORDER_DEFAULT)
    name2="normal_thresholding"+'.jpg'
    s1=s1+1
    #cv2.imwrite(name2,thr)
  ##  #cv2.waitKey(10)
    #img=cv2.equalizeHist(img)
    #cv2.normalize(img,img,0,255,cv2.NORM_MINMAX)
    c_part_size=1
    r_part_size=1
  ##  cv2.imshow("img1",img)
    ###@@@@@@cv2.waitKey(0)
    element1=cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    element2=cv2.getStructuringElement(cv2.MORPH_CROSS,(1,3))
    thr=cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,1)
    #_,thr=cv2.threshold(img,50,255,cv2.THRESH_BINARY_INV)
    cv2.imshow('thr.jpg',thr)  
    ##cv2.waitKey(0)
    #thr= cv2.morphologyEx(thr, cv2.MORPH_OPEN, element1,5) ###not a good ideA as chars merge
   ## cv2.imshow("thr",thr)
    
    element_x=np.ones((2,1),np.uint8)
    element_y=np.ones((1,2),np.uint8)
    t=sauvola_threshold(img) ###sauvola thresholding
    median=np.median(img)
    lower=min(0,(1-.1)*median)
    upper=min(255,(1+.1)*median)
    #canny = cv2.Canny(img,lower,upper,apertureSize = 3)
    median=np.median(img)
    #print median
    canny = cv2.Canny(img,median*0.8,median*1.1,apertureSize = 3)
    #canny = cv2.Canny(img,t*0.5,t,apertureSize = 3)
    temp=np.zeros((rows,cols),np.uint8)

    for x3 in range(0,rows,r_size):
        for y3 in range(0,cols,r_size):
            ##print x,y
            if x3>rows or y3>cols:
                continue
            if y3+r_size>cols:
                sliced_img=img[x3:x3+r_size,y3:cols]
                temp[x3:x3+r_size,y3:cols]=cv2.Canny(sliced_img,t[i,j]*0.33,t[i,j],apertureSize = 3)
            if x3+r_size>rows:
                sliced_img=img[x3:rows,y3:y3+r_size]
                temp[x3:rows,y3:y3+r_size]=cv2.Canny(sliced_img,t[i,j]*0.33,t[i,j],apertureSize = 3)
            else:
                sliced_img=img[x3:x3+r_size,y3:y3+r_size]
                temp[x3:x3+r_size,y3:y3+r_size]=cv2.Canny(sliced_img,t[i,j]*0.33,t[i,j],apertureSize = 3)
            j=j+1
        j=0
        i=i+1
    #canny=temp
    
    cv2.imshow('canny.jpg',canny)  
    ##cv2.waitKey(0)
    thr1= cv2.morphologyEx(canny, cv2.MORPH_OPEN, element_x,5)
    cv2.imshow('thr1.jpg',thr1)  
    ##cv2.waitKey(0)
    sobelx = cv2.Sobel(thr1,cv2.CV_8UC1,1,0,ksize=5)
                    
    thr2= cv2.morphologyEx(canny, cv2.MORPH_OPEN, element_y,5)
    cv2.imshow('thr2.jpg',thr2)  
    ##cv2.waitKey(0)
    #hr2= cv2.morphologyEx(thr2, cv2.MORPH_ERODE,element_y,5)
    sobely = cv2.Sobel(canny,cv2.CV_8UC1,0,1,ksize=3)
      
    sobel=cv2.bitwise_or(sobelx,sobely)
    cv2.imshow('sobelx.jpg',sobelx)  
    ##cv2.waitKey(0)
    cv2.imshow('sobely.jpg',sobely)  
    ##cv2.waitKey(0)
    #cv2.imwrite('sobel.jpg',sobel)  
    ##cv2.waitKey(0)
    contoursy,hierarchy=cv2.findContours(sobely.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    contoursx,hierarchy=cv2.findContours(sobelx.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)



    ###variables for contours
    count=0
    y1=[0]*2000
    x2=[0]*2000
    x1=[0]*2000
    y2=[0]*2000
    h2=[0]*2000
    h=[0]*2000
    w=[0]*2000
    area=[0]*200
    w1=[0]*200
    mean_x=0
    hrzntl1=0
    hrzntl2=0
    hght1=0
    hght2=0
    i=0
    for i in range(len(contoursx)):
        x1[i],y1[i],_,h[i] = cv2.boundingRect(contoursx[i])
    hght1=h.index(max(h))
    h[hght1]=0
    hght2=h.index(max(h))

    lp_width=(max_x-min_x)
   # #cv2.imshow('sobely1.jpg',sobely)
    ####@@@@@cv2.waitKey(0)
    
    for i in range(len(contoursx)):
        #if i !=hght1 and i!=hght2:
        cv2.drawContours(vertical,contoursx,i,(255,255,255),-1)
    #cv2.imwrite('vertical.jpg',vertical)
    #cv2.waitKey(0)
    ##########        ##########morphology
    element_close=np.ones((3,30),np.uint8)
    element_open=np.ones((7,3),np.uint8)
    
    vertical= cv2.morphologyEx(vertical, cv2.MORPH_CLOSE, element_close,2)
    vertical= cv2.morphologyEx(vertical, cv2.MORPH_OPEN, element_open,1)
    vertical= cv2.morphologyEx(vertical, cv2.MORPH_CLOSE, element_close,2)
    
    lp=vertical
    
    
   ## #cv2.imshow('lp.jpg',lp)
    ####@@@@@cv2.waitKey(0)
    i=0
    
    contours,_=cv2.findContours(lp.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    y_f=[0]*100
    h_f=[0]*100
    x_f=[0]*100
    w_f=[0]*100
    
    
    for i in range(len(contours)):
        #x_f[i],y_f[i],w_f[i],h_f[i],_ = cv2.minAreaRect(contours[i])
        #(x_f[i],y_f[i]),(w_f[i],h_f[i]),_ = cv2.minAreaRect(contours[i])
        x_f[i],y_f[i],w_f[i],h_f[i]=cv2.boundingRect(contours[i])
        ##print rect
        #box = cv2.boxPoints(rect)
        #x_f[i],y_f[i],w_f[i],h_f[i] = np.int0(box)
        #area[i]=w_f[i]*h_f[i]
        area[i]=cv2.moments(contours[i])['m00']
    if len(contours)==0:continue
    max_area=area.index(max(area))
    cont_x=x_f[max_area]
    cont_y=y_f[max_area]
    cont_w=w_f[max_area]
    cont_h=h_f[max_area]
    ##print cont_x,cont_y,cont_w,cont_h
    
        
    for i in range(len(contours)):        
        if i ==max_area:continue
        if (y_f[i]+h_f[i]<y_f[max_area]+h_f[max_area]+10 and  y_f[i]>y_f[max_area]-10):
            ##print cont_x,cont_y,cont_w,cont_h
            cont_y1=find_lh(y_f[i]+h_f[i],cont_y+cont_h,1)
            cont_x1=find_lh(x_f[i]+w_f[i],cont_x+cont_w,1)
            cont_x=find_lh(x_f[i],cont_x,0)
            cont_y=find_lh(y_f[i],cont_y,0)
            cont_h=cont_y1-cont_y
            cont_w=cont_x1-cont_x
    
    
    extracted_img=vertical[cont_y-1:cont_y+cont_h+5,cont_x:cont_x+cont_w+5]
    thr4=thr[cont_y-1:cont_y+cont_h+5,cont_x:cont_x+cont_w+5]
    #extracted_img=np.multiply(extracted_img,255)
    img_cont=img[cont_y-1:cont_y+cont_h+5,cont_x:cont_x+cont_w+5]
    img1_cont=img1[cont_y-1:cont_y+cont_h+5,cont_x:cont_x+cont_w+5]
    color = plate_color(img1_cont)
    final=cv2.bitwise_and(extracted_img,thr4)
    final=cv2.dilate(final,np.ones((3,3),np.uint8),2)
    final=cv2.erode(final,np.ones((3,3),np.uint8),1)
    cv2.imshow("final1.jpg",final)
    '''
    #img1 = cv2.equalizeHist(img1)
    img1_cont= img1[cont_y:cont_y+cont_h,cont_x:cont_x+cont_w]
    hsv = cv2.cvtColor(img1_cont,cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(hsv)
    
    #print h.max()
    #print h.min()
    hist = cv2.calcHist([h],[0],None,[256],[0,256])            
    plt.plot(range(hist.size),hist)
    hist1 = cv2.calcHist([s],[0],None,[256],[0,256])            
    plt.plot(range(hist1.size),hist1)
    hist2 = cv2.calcHist([v],[0],None,[256],[0,256])            
    plt.plot(range(hist2.size),hist2)
    plt.show()
    '''
    
    '''
    for a in final1:
        b=np.trim_zxeros(a)
        final2[i]=
    '''
    ##print type(final1)
    ##print type(final2)
    #final=cv2.bitwise_and(final,thr4)
    #extracted_thr=cv2.adaptiveThreshold(final1,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,15,1)
    #extracted_img =[np.trim_zeros(np.array(a)) for a in extracted_img]
    #_,extracted_thr=cv2.threshold(final1,50,255,cv2.THRESH_BINARY_INV)
    #cv2.imwrite("final2.jpg",final)
    #cv2.waitKey(0)
    #name=f[len(path)+1:len(f)]+"contour_"+".jpg"
    dst=path+"/"+"output"+"/"+f+"/"
    #os.mkdir(dst,0777)
    
    
    s1=s1+1
    #####noise removal######## # # # # #
    #final=extracted_thr
    char_cont,hierarchy=cv2.findContours(final.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    h=[0]*len(char_cont)
    x=[0]*len(char_cont)
    y=[0]*len(char_cont)
    w=[0]*len(char_cont)
    noise_x=[]
    noise_y=[]
    noise_h=[]
    noise_w=[]
    #char_cont1=[0]*len(char_cont)
    aspect_ratio=[]
    for i in range(len(char_cont)):
        #cv2.drawContours(char_cont1[i],contours,i,[255,255,255],-1)
        x[i],y[i],w[i],h[i],=cv2.boundingRect(char_cont[i])
        cv2.imshow('char_cont',final[y[i]:y[i]+h[i],x[i]:x[i]+w[i]])
        
        #cv2.waitKey(0)
        aspect_ratio.append(w[i]/float(h[i]))
    ##print aspect_ratio
    chars=[0]*50
    svm_label=[0]*50
    #print "first time svm calling"
    chars,svm_label,cal_aspr,_,_,_,_=svm_calc(None,final)
    
    svm_calling=svm_calling+1
    #print "svm_calling"
    #print svm_calling
    final_chars=[]
    
    i=0
    if cal_aspr!=None:
    
        for ind in range(len(char_cont)):
            solidity=cv2.moments(char_cont[ind])['m00']
            
        
            if (aspect_ratio[ind] in cal_aspr):
                if solidity>100 and ((svm_label[ind]==1 or svm_label[ind]==2)):
            
                    #print 'aspect_ratio'+ str(aspect_ratio[ind])
                    
                    final_chars.append(chars[ind])
                    cv2.imshow("check x",chars[ind])
                   # cv2.waitKey(0)
                    #print "x="+str(x[ind])
                    noise_x.append(x[ind])
                    noise_y.append(y[ind])
                    noise_h.append(h[ind])
                    noise_w.append(w[ind])
            else:
                if h[ind]>10 and w[ind]>10:# and x[i]>5 and y[i]>5:
                    char=img_cont[y[ind]:y[ind]+h[ind],x[ind]:x[ind]+w[ind]]
                    x[ind],y[ind],w[ind],h[ind]=cv2.boundingRect(char_cont[ind])
                    ##print char_cont[ind]
                    sub=np.zeros(img.shape,np.uint8)
                    cv2.drawContours(sub,char_cont,ind,(255,255,255),-1)
                    temp=sub[y[ind]:y[ind]+h[ind],x[ind]:x[ind]+w[ind]]
                     
                    thr=cv2.adaptiveThreshold(char,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,3)
                    sub3=cv2.bitwise_and(thr,temp)
                   ## cv2.imshow("thr",thr)
                   ## cv2.waitKey(0)
                    #img0=thr
                    kernel=np.ones((3 ,1),np.uint8)
                    img0=cv2.erode(sub3,kernel,15)
                    #img0=cv2.morphologyEx(sub3,cv2.MORPH_OPEN,kernel,1)
                    #kernel=np.ones((1 ,3),np.uint8)
                    #img0=cv2.morphologyEx(img0,cv2.MORPH_CLOSE,kernel,1)
                    #print ind
                   ## cv2.imshow("deleted char",img0)
                 ##   cv2.waitKey(0)
                    #print "deleted_char"+str(ind)
                    #del chars[ind]
                    #print "second time svm calling"
                    
                    #del svm_label[ind]
                    
                    char_temp,label_temp,_,w_small,x_small,y_small,h_small=svm_calc(chars[ind],img0)
                
                    if char_temp!=None:
                        for i in range(len(char_temp)):
                            #print i
                            cv2.imshow("char_temp",char_temp[i])
                            #print label_temp[i]
                            ##cv2.waitKey(0)
                        #####char size comparison
                    
                    
                        height=max(h_small)
                        noise=min(h_small)
                        noise_index=h_small.index(noise)
                            
                        if noise<2*height/3:
                            del char_temp[noise_index]
                            del label_temp[noise_index]
                            del x_small[noise_index]
                            del y_small[noise_index]
                            del h_small[noise_index]
                            del w_small[noise_index]

                        check=0
                        count=0              
                        temp0=char_temp[0].shape[0]
                        temp1=char_temp[0].shape[1]
                        #print temp0,temp1
                        asp_ratio=temp1/float(temp0)
                        #print "asp_ration"+str(asp_ratio)+","+str(comp1)
                        for j in range(1,len(char_temp)):
                            if (char_temp[j].shape[1]/float(temp1)>.75 or char_temp[j].shape[1]/float(temp1)<.5) or (abs(asp_ratio-comp1)>.3):#(abs(char_temp[j].shape[0]-temp0)>3 or char_temp[j].shape[1]/float(temp1)>.75 or char_temp[j].shape[1]/float(temp1)<.5) and 
                                # or (char_temp[j].shape[1]/float(temp1)<2 or char_temp[j].shape[1]/float(temp1)>1.333) :#and (char_temp[j].shape[1]/char_temp[j].shape[0]-comp)<.4:
                                #if (asp_ratio-comp1)>.3
                                    #print char_temp[j].shape[1],float(temp1)
                                    check=check+1
                                    #break
                        if check==1 and (svm_label[i]==1 or svm_label[i]==2):
                            final_chars.append(chars[ind])
                            noise_x.append(x[ind])
                            noise_y.append(y[ind])
                            noise_h.append(h[ind])
                            noise_w.append(w[ind])
                            #print"check=1 wala"
                          ##  cv2.imshow("check",chars[i])
                         ##   cv2.waitKey(0)
                        else:
                            #print"check=0 wala"
                            ##print label_temp
                            if char_temp != None:
                                #print "label_temp"
                                #print label_temp
                                for k in range(len(char_temp)):
                                    if label_temp[k]==1 or label_temp[k]==2:# or label_temp[k]==-1:
                                        final_chars.append(char_temp[k])
                                
                                        
                                        noise_x.append(x[ind]+x_small[k])
                                        if y_small[k]>y[ind]:
                                            noise_y.append(y[ind]+y_small[k])
                                        else:
                                            noise_y.append(y[ind])
                                        noise_h.append(h_small[k])
                                        noise_w.append(w_small[k])
        
                                               
    s2=1
    
    #file1.write(f[len(path):len(f)]+"    ")
    lp_folder = path+"/"+"output"+"/"+f[len(path)+1:len(f)-4]+'/'
    if os.path.isdir(lp_folder)==False:
        os.mkdir(lp_folder,0777)
    size1=256
    final_chars2=cal_height(final_chars,noise_x,noise_w,noise_h)
    ##print final_chars2
    if final_chars2!=None and len(final_chars2)!=0:
        name_char = [0]*len(final_chars2)
        #file1.write(f[len(path):len(f)]+"    ")
        for i in xrange(len(final_chars2)-1,-1,-1):
            size_h=2*final_chars2[i].shape[0]
            size_w=2*final_chars2[i].shape[1]
            sub=cv2.resize(final_chars2[i],(size_w,size_h),2,2,cv2.INTER_AREA)
            if sub.shape[0]<size1 and sub.shape[1]<size1:
                #print f[len(path)+1:len(f)-4]
                last=cv2.copyMakeBorder(sub,(size1-sub.shape[0])/2,(size1-sub.shape[0])/2,(size1-sub.shape[1])/2,(size1-sub.shape[1])/2,cv2.BORDER_CONSTANT,value=[0,0,0])
                name_char[i]=path+"/"+"output"+"/"+f[len(path)+1:len(f)-4]+"/"+"char_"+str(s2)+".jpg"
                cv2.imwrite(name_char[i],last)
                s2=s2+1
        len(name_char)
        output1 = deep_classify(name_char)
        file1.write(output1+',')
    if color ==1:
		file1.write('  YELLOW ')
    else:
        file1.write('  WHITE ')
	file1.write('\n')
	svm_calling=0
    '''
    print np.char.strip(f)
    print np.char.strip(f1)
    while np.char.strip(f) == np.char.strip(f1):
        input_path.close()
        input_path = open('input.txt','r')
        f1 = input_path.readline() 
        print 'WAITING'
        time.sleep(.001)
        comp_count=comp_count+1
        if comp_count==3000:
            loop=1
    '''
    comp_count=0
    loop=1
	

file1.close()        
cv2.destroyAllWindows()