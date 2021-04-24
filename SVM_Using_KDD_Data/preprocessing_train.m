 
clc
close all


str = fileread('kddcupdata10.txt');
str = strrep( str, 'S0', 'SG' );
str = strrep( str, 'normal.', 48 );
str = strrep( str, 'neptune.',49 );
str = strrep( str, 'back.',49 );
str = strrep( str, 'land.',49 );
str = strrep( str, 'pod.',49 );
str = strrep( str, 'smurf.',49 );
str = strrep( str, 'teardrop.',49 );
str = strrep( str, 'satan.',50 );
str = strrep( str, 'ipsweep.',50 );
str = strrep( str, 'nmap.',50 );
str = strrep( str, 'portsweep.',50 );
str = strrep( str, 'guess_passwd.',51 );
str = strrep( str, 'ftp_write.',51 );
str = strrep( str, 'imap.',51 );
str = strrep( str, 'phf.',51 );
str = strrep( str, 'multihop.',51 );
str = strrep( str, 'warezmaster.',51 );
str = strrep( str, 'warezclient.',51 );
str = strrep( str, 'spy.',51 );
str = strrep( str, 'buffer_overflow.',52 );
str = strrep( str, 'loadmodule.',52 );
str = strrep( str, 'perl.',52 );
str = strrep( str, 'rootkit.',52 );

data =textscan(str, '%s','Delimiter','');

data = cat(1,data{:});
%data{55086,1}=[];
String_data = regexp(data,'([A-Z a-z  _ ]+)','match'); % Remove all numeric
Numeric_data = regexp(data,'([0.00000-9.99999 10-200000]+)','match'); % Remove all letters
for i=1:size(data,1)
    if size(Numeric_data{i,1},2)~=39 || size(String_data{i,1},2)~=3
        Numeric_data{i,1}=[];
        String_data{i,1}=[];
    end
    
end
emptyCells = cellfun('isempty', Numeric_data); 

Numeric_data(all(emptyCells,2),:) = [];

Numeric_data = cat(1,Numeric_data{:});
String_data = cat(1,String_data{:});
DATA=[Numeric_data(:,1) Numeric_data(:,2:end)];
training_data=cellfun(@str2num,DATA); 
training_data=unique(training_data,'rows');
features=training_data(:,1:end-1);
train_class=training_data(:,end);
%save('features')
save('train_class')
%train_feature=load('features');
train_feature=zscore(features);

save('train_feature');
save('train_feature.mat', 'train_feature')
save('train_class.mat', 'train_class')
%label=load('class');

%modelSVM = svmtrain(train_class, train_feature, ' -q -s 0 -t 0 -c 2.00000000');