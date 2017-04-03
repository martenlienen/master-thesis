function [flowx, flowy] = compute_flow_field(events_path, window_length, nrows, ncols)

  data = csvread(events_path, 1, 0);
  ts = data(:, 1);
  x = data(:, 2);
  y = data(:, 3);
  pol = data(:, 4);

  tmin = min(ts);
  nwindows = ceil((max(ts) - tmin) / window_length);

  flowx = cell(nwindows, 1);
  flowy = cell(nwindows, 1);

  for i = 1:nwindows
    % We use sliding window of double the actual window size so that the flow
    % field for the first events in the window can be fitted to events from the
    % previous window.
    tstart = binsearch(ts, (i - 2) * window_length + tmin);
    toffset = binsearch(ts, (i - 1) * window_length + tmin) - tstart + 1;
    tend = binsearch(ts, i * window_length + tmin);
    tflt = tstart:tend;

    [vx, vy] = computeFlow(x(tflt), y(tflt), ts(tflt), pol(tflt), 5, window_length, 0.1, ncols, nrows, toffset);

    flowx{i} = sparse(vx);
    flowy{i} = sparse(vy);

    fprintf('Iteration %d / %d\n', i, nwindows);
    drawnow('update');
  end

end

function idx = binsearch(arr, needle)

  a = 1;
  b = size(arr, 1);

  while (a < b)
    idx = floor((a + b) / 2);

    if (needle > arr(idx))
      a = idx + 1;
    elseif (needle < arr(idx))
      b = idx - 1;
    else
      a = idx;
      b = idx;
    end
  end

  idx = a;

end
