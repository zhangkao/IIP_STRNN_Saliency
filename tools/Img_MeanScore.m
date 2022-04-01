function Img_MeanScore(ResDir)
 
    if nargin < 1
        ResDir = fullfile('E:\DataSet\salicon\val\Results_keras\');
    end
	
    scoreDir = [ResDir 'Scores' filesep];
    outDir   = ResDir;

    d = dir([scoreDir, '*.mat']);
    imgFiles = {d(~[d.isdir]).name};
    fileNum = length(imgFiles);

    meanS = struct;
    for i = 1:fileNum
        temName = [scoreDir imgFiles{i}];
        S = load(temName);
        S = S.scores;

        S(isnan(S)) = 0;
        tms = mean(S);

        meanS(i).name= imgFiles{i}(7:end-4);
        meanS(i).AUC_S  = tms(1);
        meanS(i).NSS    = tms(2);
        meanS(i).AUC_J  = tms(3);
        meanS(i).AUC_B  = tms(4);
        meanS(i).KL     = tms(5);
        meanS(i).SIM    = tms(6);
        meanS(i).CC     = tms(7);
        meanS(i).scores = S;
    end
    save([outDir 'meanS_mat.mat'],'meanS');

end
