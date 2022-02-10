
import cv2
import numpy as np
# np.random.seed(<int>) # you can use this line to set the fixed random seed if you are using np.random
import random
# random.seed(123) 
# you can use this line to set the fixed random seed if you are using random
import pandas as pd

def homography_calc(dist):
    A=[]
    for item in dist:
        x=item[0][0]
        y=item[0][1]
        xdash=item[1][0]
        ydash=item[1][1]
        A.append([x,y,1,0,0,0,-xdash*x,-xdash*y,-xdash])
        A.append([0,0,0,x,y,1,-ydash*x,-ydash*y,-ydash])
    A=np.asarray(A)    
    U,S,VT=np.linalg.svd(A)
    h=VT[-1]
    H=(h/h[-1]).reshape(3,3)

    return H

def solution(left_img, right_img):
    r_rows=right_img.shape[0]
    r_cols=right_img.shape[1]
    l_rows=left_img.shape[0]
    l_cols=left_img.shape[1]
    
    sift=cv2.SIFT_create()
    #MARK POINTS OF INTEREST FOR LEFT SIDE IMAGE
    left_img_gray = cv2.cvtColor(left_img,cv2.COLOR_BGR2GRAY)    
    left_poi, left_des = sift.detectAndCompute(left_img_gray,None)
    # img=cv2.drawKeypoints(left_img_gray,left_poi,left_img,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # cv2.imwrite('left_img_poi.jpg', img)

    #MARK POINTS OF INTEREST FOR RIGHT SIDE IMAGE
    right_img_gray=cv2.cvtColor(right_img,cv2.COLOR_BGR2GRAY)
    right_poi, right_des = sift.detectAndCompute(right_img_gray,None)
    # img=cv2.drawKeypoints(right_img_gray,right_poi,right_img,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # cv2.imwrite('right_img_poi.jpg', img)
    # print(len(left_poi),len(right_poi))
    dist=[]
    # rkp=[]
    eta=0.75
    r=np.asarray(right_des)
    for i in range(len(left_des)):
        l=left_des[i]
        data={'eucl':np.linalg.norm(l-r,axis=1)}
        distance_df=pd.DataFrame(data)
        distance_df['right']=pd.Series(list(r))
        distance_df['rkp']=pd.Series(list(right_poi))
        distance_df=distance_df.sort_values(['eucl'],ascending=True).reset_index()
        qm=distance_df['rkp'][0].pt
        qn=distance_df['rkp'][1].pt
        pqm=distance_df['eucl'][0]
        pqn=distance_df['eucl'][1]
        if pqm<eta*pqn:
            dist.append((left_poi[i].pt,qm))
            # rkp.append(qn)
    # print(len(dist))
    #RANSAC
    n=4
    k=6000
    t=100
    in_count=0
    for c in range(k):        
        sample=random.sample(dist,n) 
        pidash=[]
        pi=[]
        for d in dist:
            if d not in sample:
                pidash.append(d[0])
                pi.append(d[1])
                  
        H=homography_calc(sample)
        inliers=[]
        for i in range(len(pi)):    
            lst1=list(pidash[i])
            lst1.append(1)
            lst2=list(pi[i])
            lst2.append(1)
            left_tup=tuple(lst1)
            right_tup=tuple(lst2)   
            Hpi=np.dot(H,right_tup)
            Hpi=Hpi/Hpi[-1]
            if np.linalg.norm(left_tup - Hpi) < t:
                # print('found inliers')
                inliers.append((pidash[i],pi[i]))
        if in_count<len(inliers):
            in_count=len(inliers)
            largest_set_inliers=inliers
    
    new_H=homography_calc(largest_set_inliers)
    
    # print(in_count)

    left_reshaphed=np.float32([[0,0],[0,r_rows],[r_cols,r_rows],[r_cols,0]])
    
    left_reshaphed=left_reshaphed.reshape(-1,1,2)
    # print(left_reshaphed.shape)
    right_reshaped=np.float32([[0,0],[0,l_rows],[l_cols,l_rows],[l_cols,0]])
    right_reshaped=right_reshaped.reshape(-1,1,2)

    pers_transformed=cv2.perspectiveTransform(src=right_reshaped,m=new_H)
    # print(pers_transformed.shape,pers_transformed)
    f=np.append(pers_transformed,left_reshaphed,axis=0)
    # print(f.shape)
    min_values=np.min(f,axis=0).ravel()
    # print(type(min_values),min_values[0])
    x_min=int(min_values[0])
    y_min=int(min_values[1])
    max_values=np.max(f,axis=0).ravel()
    x_max=int(max_values[0])
    y_max=int(max_values[1])
    offset=(x_max-x_min,y_max-y_min)
    trans=np.array([[1,0,-x_min],[0,1,-y_min],[0,0,1]])
    # print(trans)
    result_img=cv2.warpPerspective(src=left_img,M=np.dot(trans,new_H),dsize=(offset))
    result_img[-y_min:r_rows-y_min,-x_min:r_cols-x_min]=right_img

    return result_img
    

if __name__ == "__main__":
    left_img = cv2.imread('left.jpg')
    right_img = cv2.imread('right.jpg')
    result_img = solution(left_img, right_img)
    cv2.imwrite('results/task1_result.jpg', result_img)


