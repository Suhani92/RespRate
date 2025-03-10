load hychirp
plot(t,hychirp)
grid on
title("Signal")
axis tight
xlabel("Time (s)")
ylabel('Amplitude')
view([-40 30])

function helperPlotScalogram3d(sig,Fs)
% This function is only intended to support this wavelet example.
% It may change or be removed in a future release.
figure
[cfs,f] = cwt(sig,Fs);

sigLen = numel(sig);
t = (0:sigLen-1)/Fs;
surface(t,f,abs(cfs));
xlabel("Time (s)")
ylabel("Frequency (Hz)")
zlabel("Magnitude")
title("Scalogram In 3-D")
set(gca,Yscale="log")
shading interp
view([-40 30])
end

helperPlotScalogram3d(hychirp,Fs)