load('scan_medial.mat')
data = TDB1012FTDB101ResultsandProcessedData2FTDB101TopoData2FMatrixFi;

strat = NaN(size(data));
strat(:, end) = data(:, 1);
for i = size(data,2)-1:-1:1
    strat(:, i) = min([data(:, i+1), data(:, i)], [], 2);
    
end

dz = data(:, 2:end) - data(:, 1:end-1);
figure()
histogram(dz(:))



if true
    figure()
    hold on
    cmap = repelem(lines(10), round(size(data, 2), -1)/10, [1 1 1]);
    cmap = jet(size(data, 2));
    x = 1:size(data, 1);
    for i = 2:size(data,2)-1
    %     plot(x, strat(:, i), 'Color', cmap(i, :))
        patch([x, fliplr(x)], [strat(:, i)' fliplr(strat(:, i-1)')], cmap(i, :))
        drawnow

    end
end