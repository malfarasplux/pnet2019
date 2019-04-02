function get_sepsis_score(input_zip_file, output_zip_file)
    % get input files
    input_files = sort(unzip(input_zip_file, 'tmp_inputs'));
    j = 1;
    for i = 1 : length(input_files)
        if ~isfile(input_files(j))
            input_files(j) = [];
        else
            j = j + 1;
        end
    end

    % make temporary output directory
    mkdir('tmp_outputs');

    % generate scores
    n = length(input_files);
    for i = 1:n
        % read data
        input_file = input_files{i};
        data = read_challenge_data(input_file);

        % make predictions
        [scores, labels] = compute_sepsis_score(data);
        
        % write results
        file_name = strsplit(input_file, filesep);
        output_file = ['tmp_outputs' filesep file_name{end}];

        fid = fopen(output_file, 'wt');
        fprintf(fid, 'PredictedProbability|PredictedLabel\n');
        dlmwrite(output_file, [scores labels], 'delimiter', '|', '-append');
        fclose(fid);
    end

    % perform clean-up
    zip(output_zip_file, 'tmp_outputs');
    rmdir('tmp_outputs','s');
    rmdir('tmp_inputs','s');
end

function [scores, labels] = compute_sepsis_score(data)
    x_mean = [ ...
        83.8996 97.0520  36.8055  126.2240 86.2907 ...
        66.2070 18.7280  33.7373  -3.1923  22.5352 ...
        0.4597  7.3889   39.5049  96.8883  103.4265 ...
        22.4952 87.5214  7.7210   106.1982 1.5961 ...
        0.6943  131.5327 2.0262   2.0509   3.5130 ...
        4.0541  1.3423   5.2734   32.1134  10.5383 ...
        38.9974 10.5585  286.5404 198.6777];
    x_std = [ ...
        17.6494 3.0163  0.6895   24.2988 16.6459 ...
        14.0771 4.7035  11.0158  3.7845  3.1567 ...
        6.2684  0.0710  9.1087   3.3971  430.3638 ...
        19.0690 81.7152 2.3992   4.9761  2.0648 ...
        1.9926  45.4816 1.6008   0.3793  1.3092 ...
        0.5844  2.5511  20.4142  6.4362  2.2302 ...
        29.8928 7.0606  137.3886 96.8997];
    c_mean = [60.8711 0.5435 0.0615 0.0727 -59.6769 28.4551];
    c_std = [16.1887 0.4981 0.7968 0.8029 160.8846 29.5367];

    x = data(:, 1:34);
    c = data(:, 35:40);

    [m, n] = size(x);
    [r, s] = size(c);
    x_norm = (x - repmat(x_mean, m, 1))./repmat(x_std, m, 1);
    c_norm = (c - repmat(c_mean, r, 1))./repmat(c_std, r, 1);

    x_norm(isnan(x_norm)) = 0;
    c_norm(isnan(c_norm)) = 0;

    model.beta = [ ...
        0.1806  0.0249 0.2120  -0.0495 0.0084 ...
        -0.0980 0.0774 -0.0350 -0.0948 0.1169 ...
        0.7476  0.0323 0.0305  -0.0251 0.0330 ...
        0.1424  0.0324 -0.1450 -0.0594 0.0085 ...
        -0.0501 0.0265 0.0794  -0.0107 0.0225 ...
        0.0040  0.0799 -0.0287 0.0531  -0.0728 ...
        0.0243  0.1017 0.0662  -0.0074 0.0281 ...
        0.0078  0.0593 -0.2046 -0.0167 0.1239]';
    model.rho = 7.8521;
    model.nu = 1.0389;

    xstar = [x_norm c_norm];
    exp_bx = exp(xstar*model.beta);
    l_exp_bx = (4/model.rho).^model.nu * exp_bx;

    scores = 1 - exp(-l_exp_bx);
    labels = double([scores>0.45]);
end

function data = read_challenge_data(filename)
    f = fopen(filename, 'rt');
    try
        l = fgetl(f);
        column_names = strsplit(l, '|');
        data = dlmread(filename, '|', 1, 0);
    catch ex
        fclose(f);
        rethrow(ex);
    end
    fclose(f);

    % ignore SepsisLabel column if present
    if strcmp(column_names(end), 'SepsisLabel')
        column_names = column_names(1:end-1);
        data = data(:,1:end-1);
    end
end
