close all
clear all

loadModules();

ACRaw=readRawData('rawAGISCIGTS.xlsx');
ACInterp=interpolateData(ACRaw);
ACProg=labelProgression(ACInterp);

JPRaw=readRawDataJapan('japanDataRaw.xlsx');
JPInterp=interpolateDataJapan(JPRaw);
JPProg=labelProgression(JPInterp);



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



getRegModelJapanNoVar( A, C, Q, R, INITX, INITV, JPTrain);
o=readRegCoeff();

[accJPadd,ddJPadd,NTJPadd]= nonParetoAnalysis( A,C,Q,R,INITX,INITV,o, JPTrain,JPTest,0.01,'JP Training w Add. Var.','JP Testing');



[h,p]=ttest(accAC,accJP);
if h==1
  fprintf('Mean is different, accuracy AC-JP=%f, pval=%f\n',mean(accAC-accJP),p)
  else
    fprintf('Mean is same\n')
      end

      plot(NTJP,accJP,'*b')
      hold on
      plot(NTAC,accAC,'or')
      plot(NTACadd,accACadd,'.g')
      xlabel('Number of Test per Year')
      ylabel('Accuracy')
      legend('AC','JP','JP+')
      savefig('AC-JP GEE 1run.png')
save('Japan_NonPareto_GEE.mat')
