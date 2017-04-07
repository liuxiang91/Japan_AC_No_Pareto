close all
clear all

loadModules();

ACRaw=readRawData('rawAGISCIGTS.xlsx');
ACInterp=interpolateData(ACRaw);
ACProg=labelProgression(ACInterp);

JPRaw=readRawDataJapan('japanDataRaw.xlsx');
JPInterp=interpolateDataJapan(JPRaw);
JPProg=labelProgression(JPInterp);



accuracy=[];
nt=[];
dd=[];

for i=1:33
  fprintf('*****i=%d\n',i);
[ JPTrain,JPTest ] = splitTrainTest( JPProg, 0.7 );
ratio = (size(JPTrain,1)-1)/(size(ACRaw,1)-1);
[ ACTrain,~ ] = splitTrainTest( ACProg, ratio );



[ A0 C0 Q0 R0 INITX0 INITV0 ]=initializeEM(ACTrain);
[A, C, Q, R, INITX, INITV, LL] = learn_kalman(ACTrain(2:end,3), A0, C0, Q0, R0, INITX0, INITV0,100);

getRegModel( A, C, Q, R, INITX, INITV, ACTrain);
o=readRegCoeff();

[accAC,ddAC,NTAC]= nonParetoAnalysis( A,C,Q,R,INITX,INITV,o, ACTrain,JPTest,0.01,'AC Training','JP Testing');


[ A0 C0 Q0 R0 INITX0 INITV0 ]=initializeEM(JPTrain);
[A, C, Q, R, INITX, INITV, LL] = learn_kalman(JPTrain(2:end,3), A0, C0, Q0, R0, INITX0, INITV0,100);

getRegModelJapanNoVar( A, C, Q, R, INITX, INITV, JPTrain);
o=readRegCoeff();

[accJP,ddJP,NTJP]= nonParetoAnalysis( A,C,Q,R,INITX,INITV,o, JPTrain,JPTest,0.01,'JP Training w/o Add. Var.','JP Testing');



getRegModelJapan( A, C, Q, R, INITX, INITV, JPTrain);
o=readRegCoeff();

[accJPadd,ddJPadd,NTJPadd]= nonParetoAnalysis( A,C,Q,R,INITX,INITV,o, JPTrain,JPTest,0.01,'JP Training w Add. Var.','JP Testing');

accuracy=[accuracy; [accAC, accJP,accJPadd]];
nt=[nt;[NTAC,NTJP,NTJPadd]];
dd=[dd;[ddAC,ddJP,ddJPadd]];

end

[h,p]=ttest(accuracy(:,1),accuracy(:,2));
if h==1
  fprintf('Mean is different, accuracy AC-JP=%f, pval=%f\n',mean(accuracy(:,1)-accuracy(:,2)),p)
  else
    fprintf('Mean is same\n')
      end

      [h,p]=ttest(accJP,accJPadd);
if h==1
  fprintf('Mean is different, accuracy JP-JP+=%f, pval=%f\n',mean(accuracy(:,2)-accuracy(:,3)),p)
  else
    fprintf('Mean is same\n')
      end
f=figure();
hold on
c={'.b','.g','.r'};
for i=1:3
      plot(nt(:,i),accuracy(:,i),c{i})
end
      xlabel('Number of Test per Year')
      ylabel('Accuracy')
      legend('AC','JP','JP+')
      saveas(f,'AC-JP GLM 33run.png')
save('Japan_NonPareto_GLM_33.mat')
