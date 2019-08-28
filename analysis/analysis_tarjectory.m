% for 2d - 3v
% load('fit.mat')
max(episode)
ep=find(episode==5000);

x= observation(ep,16);
y= observation(ep,17);

vx= observation(ep,18);
vy= observation(ep,19);

figure;
subplot(2,2,1)
    hold on
plot(x,y,'k-')
for ii = 1:10:length(ep)-1
    ah = annotation('arrow',...
            'headStyle','cback1',...
            'HeadLength',8,...
            'HeadWidth',5,...
            'Units','normalized');
    set(ah,'parent',gca);   
    set(ah,'position',...
        [x(ii) y(ii) ...
        0.1*vx(ii) 0.1*vy(ii)]);

end
axis([-300,300,-300,300])
title('trajectory')
subplot(2,2,2)
plot(vx,vy,'k-')
title('trajectory')

subplot(2,2,2)
hold on
plot(1:length(ep),vx,'k')
plot(1:length(ep),vy,'r')
title('speed')

subplot(2,2,3)
hold on
plot(1:length(ep),action(ep),'k')
plot(1:length(ep),action(ep),'r')
title('acclecation')

subplot(2,2,4)
hold on
plot(1:length(ep),reward(ep),'k')
title('reward')

