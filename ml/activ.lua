-- Initialization
local activ = {}




-- Activation Functions
activ.tanh = math.tanh
function activ.linear(x) return x end
function activ.expit(x) return 0x01/(0x01+math.exp(-x)) end




-- De-initialization
return activ
