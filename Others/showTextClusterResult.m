path_pic='G:/浙大实习/allImages-8337/';
path_label='G:/浙大实习/text聚类/';
path_result='G:/浙大实习/聚类结果/';
% path_pic='G:/浙大实习/text显示/';
% path_label='G:/浙大实习/text聚类/';
% path_result='G:/浙大实习/text显示_外框/';
dList=dir([path_label,'*.txt']); 
N=length(dList); 

for k=1:N
    tt=textread([path_label dList(k).name]);
    sh=imread([path_pic dList(k).name(1:length(dList(k).name)-4) '.jpg']);
    [nn mm]=size(tt);
    for i=1:nn
        for j=1:4
            if tt(i,j) < 1
                tt(i,j) = 1;
            end
        end
        sh(tt(i,1):tt(i,2),tt(i,3),1)=255;
        sh(tt(i,1):tt(i,2),tt(i,3),2)=0;
        sh(tt(i,1):tt(i,2),tt(i,3),3)=0;
        sh(tt(i,1):tt(i,2),tt(i,4),1)=255;
        sh(tt(i,1):tt(i,2),tt(i,4),2)=0;
        sh(tt(i,1):tt(i,2),tt(i,4),3)=0;
        sh(tt(i,1),tt(i,3):tt(i,4),1)=255;
        sh(tt(i,1),tt(i,3):tt(i,4),2)=0;
        sh(tt(i,1),tt(i,3):tt(i,4),3)=0;
        sh(tt(i,2),tt(i,3):tt(i,4),1)=255;
        sh(tt(i,2),tt(i,3):tt(i,4),2)=0;
        sh(tt(i,2),tt(i,3):tt(i,4),3)=0;    
    end      
    imwrite(sh,[path_result dList(k).name(1:length(dList(k).name)-4) '.jpg'],'quality',100);
    %imwrite(sh,[path_result dList(k).name(1:length(dList(k).name)-4) '.bmp']);
    %imwrite(sh,[path_result dList(k).name(1:length(dList(k).name)-4) '.jpg'],'quality',100,'mode','lossless');

end