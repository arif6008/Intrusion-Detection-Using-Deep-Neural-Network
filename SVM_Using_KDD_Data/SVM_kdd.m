clc
close all
tic
train_f=cell2mat(struct2cell(load('train_feature.mat')));
train_c=cell2mat(struct2cell(load('train_class.mat')));

model = train(train_c, sparse(train_f), '-s1 -v6');
%model.trainAcc = train(train_c, sparse(train_f), '-s1 -v6');

test_f=cell2mat(struct2cell(load('test_feature.mat')));
%test_f=zscore(test_f);
test_c=cell2mat(struct2cell(load('test_class.mat')));

disp("Training Accuracy:")
predicted_label_train = predict(train_c, sparse(train_f), model);
disp("Testing Accuracy:")
predicted_label = predict(test_c, sparse(test_f), model);
%test_accuracy=accuracy
toc