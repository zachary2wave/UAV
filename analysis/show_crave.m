
load('fit.mat')
figure
subplot(2,2,1)
plot(metrics(:,1))
xlabel('step')
ylabel('loss')
subplot(2,2,2)
plot(metrics(:,2))
xlabel('step')
ylabel('mean_absolute_error')
subplot(2,2,3)
plot(metrics(:,3))
xlabel('step')
ylabel('mean_q') 
subplot(2,2,4)
plot(reward)
title('reward')




