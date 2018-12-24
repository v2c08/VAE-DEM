function g = gx_classifier_m(x,v,P)

    % -- for debug only -- %
    %py.sys.setdlopenflags(int32(10)); % Set RTLD_NOW and RTLD_DEEPBIND
    %clear classes
    %mod = py.importlib.import_module('classifier');
    %py.importlib.reload(mod);
    % -- -- -- -- -- -- -- %
    x
    v
    pause(10)
    g = find(py.classifier.predict(full(x)).single);
    g
    pause(5)
end
