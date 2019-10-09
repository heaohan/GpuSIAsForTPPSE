% Different from furtherProcess, this file is used 
% to display the results for multiple times operation
% of heurostic optimization
clear
selpath = uigetdir(pwd, 'result folder');
indexStr = strfind(selpath, '\');
selpathUp = selpath(1:indexStr(end) - 1);

d = dir(selpath);
isub = [d(:).isdir]; %# returns logical vector
nameFolds = {d(isub).name}';
nameFolds(ismember(nameFolds,{'.','..'})) = [];

tp = zeros(length(nameFolds),1);
for i = 1:length(tp)
    tp(i) = str2double(nameFolds{i});
end
tp = sort(tp);
for i = 1:length(tp)
    nameFolds{i} = num2str(tp(i));
end

g_bests = zeros(length(nameFolds),1);
time_uses = g_bests;
deltas = zeros(length(nameFolds),1);

load([selpathUp '\' 'colors.mat']); % the colors for fitness plot; set use rand(20,3) beforehand;
% please make sure size(colors, 1) == length(nameFolds)

if (exist([selpath '\' nameFolds{i} '\' 'iter_rec.mat'], 'file') == 2)
  iter_rec_processed = cell(length(nameFolds),1);
end
fit_rec_processed = cell(length(nameFolds),1);
convergence_index = zeros(length(nameFolds),1);

figure,
for i = 1:length(nameFolds)
    load([selpath '\' nameFolds{i} '\' 'g_best.mat']);
    g_bests(i) = g_best;
    load([selpath '\' nameFolds{i} '\' 'time_use.mat']);
    time_uses(i) = time_use;
    load([selpath '\' nameFolds{i} '\' 'g_best_pos.mat']);
    deltas(i) = g_best_pos(end);
    
    load([selpath '\' nameFolds{i} '\' 'fit_rec.mat']);
    fit_rec_processed{i} = fit_rec(find(fit_rec > 0, 1):end);
    if (exist([selpath '\' nameFolds{i} '\' 'iter_rec.mat'], 'file') == 2)
        load([selpath '\' nameFolds{i} '\' 'iter_rec.mat']);
        iter_rec_processed{i} = iter_rec(find(fit_rec > 0, 1):end);
        %plot(iter_rec_processed, fit_rec_processed,'--*', 'Color', rand(1,3));
        plot(iter_rec_processed{i}, fit_rec_processed{i},'--*', 'Color', colors(i,:));
        hold on
    else
        %plot(fit_rec_processed, '--*', 'Color', rand(1,3));
        plot(fit_rec_processed{i}, '--*', 'Color', colors(i,:));
        hold on
    end
    
    % search the convergence points
    for count = length(fit_rec_processed{i}):-1:2
      if (abs(fit_rec_processed{i}(count-1) - fit_rec_processed{i}(count)) > 1e-1)
         convergence_index(i) = count;
         break;
      end
    end
end
xlabel('iteration'); ylabel('fitness'); hold off; %title('Fitness');
figure, plot(time_uses, 'r--*');xlabel('Run'); ylabel('Time / s'); 
xlim([1,20]);
if (exist([selpath '\' nameFolds{1} '\' 'iter_rec.mat'], 'file') == 2) 
  tp_mean = 0;
  for i = 1:length(nameFolds)
    tp_mean = tp_mean + time_uses(i) * ( iter_rec_processed{i}(convergence_index(i)) / 5000 ); %for DE, since the max iteration of DE is 5000
  end
  tp_mean = tp_mean / length(nameFolds);
  title([num2str(mean(time_uses)) ', mean convergence time = ' num2str(tp_mean)]);
else
  title([num2str(mean(time_uses)) ', mean convergence time = ' num2str(mean(time_uses.*(convergence_index / length(fit_rec_processed{1}))))]);
end

figure, plot(g_bests, 'r--*');xlabel('Run');ylabel('Min fitness');
xlim([1,20]);

% plot convergence points
figure,
for i = 1:length(nameFolds)
  if (exist([selpath '\' nameFolds{i} '\' 'iter_rec.mat'], 'file') == 2)
    plot(iter_rec_processed{i}, fit_rec_processed{i},'--*', 'Color', colors(i,:));
    hold on
  else
    plot(fit_rec_processed{i}, '--*', 'Color', colors(i,:));
    hold on
  end
end
for i = 1:length(nameFolds)
  if (exist([selpath '\' nameFolds{i} '\' 'iter_rec.mat'], 'file') == 2)
    plot(iter_rec_processed{i}(convergence_index(i)), ...
      fit_rec_processed{i}(convergence_index(i)),'p', 'MarkerFaceColor','red',...
      'MarkerSize',15);
     hold on
  else
    plot(convergence_index(i), fit_rec_processed{i}(convergence_index(i)),'p', 'MarkerFaceColor','red',...
      'MarkerSize',15);
    hold on
  end
end
xlabel('Iteration'); ylabel('Fitness'); hold off, %title('Fitness');

%%
figure,plot(abs(deltas), 'r--*'); xlabel('Run');
str = '\bf {$$\Delta \hat \varphi$$ / rad}';
ylabel(str,'Interpreter','latex'); hold on;
load([selpathUp '\' 'for_further.mat'], 'delta_true');
plot(delta_true * ones(length(deltas),1), 'b'); hold off;
xlim([1,20]);
