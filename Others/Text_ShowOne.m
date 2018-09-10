path='G:/浙大实习/text内部聚类/TextOne.txt';
t=textread(path);
[m,n]=size(t);
pictureW=900;
pictureH=1200;
for i=1:m
    picture=zeros(pictureH,pictureW,3);
    picture(:,:,1)=255;
    picture(:,:,2)=255;
    picture(:,:,3)=255;
    height=t(i,3);
    width=t(i,4);
    x1=floor((t(i,1)-height*0.5)*pictureH);
    x2=floor((t(i,1)+height*0.5)*pictureH);
    y1=floor((t(i,2)-width*0.5)*pictureW);
    y2=floor((t(i,2)+width*0.5)*pictureW);
    if x1==0
        x1P=1;
    end
    if y1==0
        y1P=1;
    end
    picture(x1:x2,y1:y2,1)=255;
    picture(x1:x2,y1:y2,2)=0;
    picture(x1:x2,y1:y2,3)=0;  
    imwrite(picture,['C:/Users/john/Desktop/' num2str(i-1) '.jpg'])
end