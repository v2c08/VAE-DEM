
function g = gx_encoder_m(x,v,P)
    
    % -- for debug only -- %
    %py.sys.setdlopenflags(int32(10)); % Set RTLD_NOW and RTLD_DEEPBIND
    %clear classes
    %mod = py.importlib.import_module('encoder');
    %py.importlib.reload(mod);
    % -- -- -- -- -- -- -- %
    
    x001(1,:,:,:) = reshape(x, [64, 64, 3]);
    g = py.encoder.predict(py.memoryview(x001)).single';
    
end