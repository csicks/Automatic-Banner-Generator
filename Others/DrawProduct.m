path_pic='G:/浙大实习/聚类结果/';
path_label='G:/浙大实习/text_product聚类/productCorner.txt';
[a1, a2, a3, a4, a5]=textread(path_label,'%f %f %f %f %s');
nn=size(a1);

for i=1:nn
    str=[path_pic a5{i}];
    sh=imread(str);
    
    if a1(i)<1
        a1(i)=1;
    end
    if a2(i)<1
        a2(i)=1;
    end
    if a3(i)<1
        a3(i)=1;
    end
    if a4(i)<1
        a4(i)=1;
    end

    sh(a1(i):a2(i),a3(i),1)=0;
    sh(a1(i):a2(i),a3(i),2)=255;
    sh(a1(i):a2(i),a3(i),3)=255;
    sh(a1(i):a2(i),a4(i),1)=0;
    sh(a1(i):a2(i),a4(i),2)=255;
    sh(a1(i):a2(i),a4(i),3)=255;
    sh(a1(i),a3(i):a4(i),1)=0;
    sh(a1(i),a3(i):a4(i),2)=255;
    sh(a1(i),a3(i):a4(i),3)=255;
    sh(a2(i),a3(i):a4(i),1)=0;
    sh(a2(i),a3(i):a4(i),2)=255;
    sh(a2(i),a3(i):a4(i),3)=255;     
    
    imwrite(sh,[path_pic a5{i}],'quality',100);
end
