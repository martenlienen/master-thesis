events_path = '/home/cqql/Seafile/Documents/davis/marten/events.csv';
window_millis = 30;
nrows = 180;
ncols = 240;

window_length = window_millis * 1000;
[pathstr, name, ext] = fileparts(events_path);
out_path = fullfile(pathstr, sprintf('flow-%dms.mat', window_millis));

addpath('../vendor/dvs-motion-flow');
[flowx, flowy] = compute_flow_field(events_path, window_length, nrows, ncols);

save(out_path, 'flowx', 'flowy');
