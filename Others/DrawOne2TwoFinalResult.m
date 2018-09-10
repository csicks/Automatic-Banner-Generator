pathP='G:/浙大实习/text_product聚类/One2TwoProduct.txt';
pathT='G:/浙大实习/text_product聚类/One2TwoText.txt';
p=textread(pathP);
t=textread(pathT);
[m,n]=size(t);
pictureW=1920;
pictureH=1080;
for i=1:m
    picture=zeros(pictureH,pictureW,3);
    picture(:,:,1)=255;
    picture(:,:,2)=255;
    picture(:,:,3)=255;
    heightP=p(i,3);
    widthP=p(i,4);
    x1P=floor((p(i,1)-heightP*0.5)*pictureH);
    x2P=floor((p(i,1)+heightP*0.5)*pictureH);
    y1P=floor((p(i,2)-widthP*0.5)*pictureW);
    y2P=floor((p(i,2)+widthP*0.5)*pictureW);
    if x1P<=0
        x1P=1;
    end
    if y1P<=0
        y1P=1;
    end
    picture(x1P:x2P,y1P:y2P,1)=255;
    picture(x1P:x2P,y1P:y2P,2)=0;
    picture(x1P:x2P,y1P:y2P,3)=0;
    xT=t(i,1)+p(i,1);
    yT=t(i,2)+p(i,2);
    x1T=floor((xT-t(i,3)*0.5)*pictureH);
    x2T=floor((xT+t(i,3)*0.5)*pictureH);
    y1T=floor((yT-t(i,4)*0.5)*pictureW);
    y2T=floor((yT+t(i,4)*0.5)*pictureW);
    if x1T<=0
        x1T=1;
    end
    if y1T<=0
        y1T=1;
    end
    xT1=t(i,5)+p(i,1);
    yT1=t(i,6)+p(i,2);
    x1T1=floor((xT1-t(i,7)*0.5)*pictureH);
    x2T1=floor((xT1+t(i,7)*0.5)*pictureH);
    y1T1=floor((yT1-t(i,8)*0.5)*pictureW);
    y2T1=floor((yT1+t(i,8)*0.5)*pictureW);
    if x1T1<=0
        x1T1=1;
    end
    if y1T1<=0
        y1T1=1;
    end  
    picture(x1T:x2T,y1T:y2T,1)=0;
    picture(x1T:x2T,y1T:y2T,2)=255;
    picture(x1T:x2T,y1T:y2T,3)=255;
    picture(x1T1:x2T1,y1T1:y2T1,1)=0;
    picture(x1T1:x2T1,y1T1:y2T1,2)=255;
    picture(x1T1:x2T1,y1T1:y2T1,3)=0;
    imwrite(picture,['C:/Users/john/Desktop/' num2str(i-1) '.jpg'])
end
