type=1;

if type==1   
    path='G:/浙大实习/text_product聚类/resultOne2OneAngleCenter.txt';
    tt=textread(path);
    [nn mm]=size(tt);
    figure(1);
    rectangle('Position',[-1,-1,2,2],'Curvature',[1,1]);
    hold on
    for i=1:nn
        tt(i,3)=tt(i,3);
        plot([0,cos(tt(i,3))],[0,sin(tt(i,3))],'r-');
        hold on
        text(cos(tt(i,3)),sin(tt(i,3)),num2str(i-1));
        hold on
    end
    hold off
    
elseif type ==2
    path='G:/浙大实习/text_product聚类/resultOne2TwoAngleCenter1.txt';
    tt=textread(path);
    [nn mm]=size(tt);
    subplot(1,2,1);
    rectangle('Position',[-1,-1,2,2],'Curvature',[1,1]);
    hold on
    for i=1:nn
        tt(i,3)=tt(i,3);
        plot([0,cos(tt(i,3))],[0,sin(tt(i,3))],'r-');
        hold on
        text(cos(tt(i,3)),sin(tt(i,3)),num2str(i-1));
        hold on
    end
    hold off
    path='G:/浙大实习/text_product聚类/resultOne2TwoAngleCenter2.txt';
    tt=textread(path);
    [nn mm]=size(tt);
    subplot(1,2,2);
    rectangle('Position',[-1,-1,2,2],'Curvature',[1,1]);
    hold on
    for i=1:nn
        tt(i,3)=tt(i,3);
        plot([0,cos(tt(i,3))],[0,sin(tt(i,3))],'r-');
        hold on
        text(cos(tt(i,3)),sin(tt(i,3)),num2str(i-1));
        hold on
    end
    hold off

    
end
 
 