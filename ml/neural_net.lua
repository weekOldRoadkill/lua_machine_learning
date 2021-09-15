-- Initialization
local neural_net = {}




-- Imports
local activ = require("ml.activ")




-- Neural Net Class
neural_net.in_size = 0x00
neural_net.out_size = 0x00

neural_net.hidden_size = 0x00
neural_net.hidden_count = 0x00

neural_net.biases = {}
neural_net.weights = {}

neural_net.func = activ.linear

neural_net.__index = neural_net




-- Neural Net Constructor Function
function neural_net.new(in_size, out_size, hidden_size, hidden_count, random_values, func)


    -- Argument Checking
    assert(type(in_size) == "number", "\"in_size\" is of wrong type")
    assert(in_size >= 0x01, "\"in_size\" is of invalid value")
    assert(type(out_size) == "number", "\"out_size\" is of wrong type")
    assert(out_size >= 0x01, "\"out_size\" is of invalid value")
    assert(type(hidden_size) == "number", "\"hidden_size\" is of wrong type")
    assert(hidden_size >= 0x01, "\"hidden_size\" is of invalid value")
    assert(type(hidden_count) == "number", "\"hidden_count\" is of wrong type")
    assert(hidden_count >= 0x01, "\"hidden_count\" is of invalid value")
    assert(type(random_values) == "boolean", "\"random_values\" is of wrong type")
    assert(type(func) == "function", "\"func\" is of wrong type")


    -- Initialization
    local new = {}

    new.in_size = in_size
    new.out_size = out_size

    new.hidden_size = hidden_size
    new.hidden_count = hidden_count

    new.func = func


    -- Biases
    new.biases = {}
    local bias_size = hidden_size*hidden_count+out_size

    if random_values then
        for i = 0x01, bias_size do new.biases[i] = math.random()*0x02-0x01 end
    else
        for i = 0x01, bias_size do new.biases[i] = 0.0 end
    end


    -- Weights
    new.weights = {}
    local weight_size = (in_size+out_size)*hidden_size+(hidden_count-0x01)*hidden_size^0x02

    if random_values then
        for i = 0x01, weight_size do new.weights[i] = math.random()*0x02-0x01 end
    else
        for i = 0x01, weight_size do new.weights[i] = 0.0 end
    end


    -- De-initialization
    setmetatable(new, neural_net)
    return new
end




-- Neural Net Run Function
function neural_net.run(self, input)
    

    -- Argument Checking
    assert(getmetatable(self) == neural_net, "\"self\" is of wrong type")
    assert(type(input) == "table", "\"input\" is of wrong type")
    assert(#input == self.in_size, "\"input\" is of wrong length")


    -- Initialization
    local hidden = {}
    local output = {}
    local bias_count = 0x01
    local weight_count = 0x01


    -- First Hidden Layer
    hidden[0x01] = {}

    for i = 0x01, self.hidden_size do
        hidden[0x01][i] = self.biases[bias_count]
        bias_count = bias_count+0x01

        for j = 0x01, self.in_size do
            hidden[0x01][i] = hidden[0x01][i]+input[j]*self.weights[weight_count]
            weight_count = weight_count+0x01
        end

        hidden[0x01][i] = self.func(hidden[0x01][i])
    end


    -- Other Hidden Layers
    for i = 0x02, self.hidden_count do
        hidden[i] = {}

        for j = 0x01, self.hidden_size do
            hidden[i][j] = self.biases[bias_count]
            bias_count = bias_count+0x01

            for k = 0x01, self.hidden_size do
                hidden[i][j] = hidden[i][j]+hidden[i-0x01][k]*self.weights[weight_count]
                weight_count = weight_count+0x01
            end

            hidden[i][j] = self.func(hidden[i][j])
        end
    end


    -- Output
    for i = 0x01, self.out_size do
        output[i] = self.biases[bias_count]
        bias_count = bias_count+0x01

        for j = 0x01, self.hidden_size do
            output[i] = output[i]+hidden[#hidden][j]*self.weights[weight_count]
            weight_count = weight_count+0x01
        end

        output[i] = self.func(output[i])
    end


    -- De-initialization
    return output
end




-- Neural Net Copy Function
function neural_net.copy(self, mut_chance, mut_amount)


    -- Argument Checking
    assert(getmetatable(self) == neural_net, "\"self\" is of wrong type")
    assert(type(mut_chance) == "number", "\"mut_chance\" is of wrong type")
    assert(mut_chance >= 0.0 and mut_chance <= 1.0, "\"mut_chance\" is of invalid value")
    assert(type(mut_amount) == "number", "\"mut_amount\" is of wrong type")
    assert(mut_amount >= 0.0, "\"mut_amount\" is of invalid value")


    -- Initialization
    local new = neural_net.new(
        self.in_size,
        self.out_size,
        self.hidden_size,
        self.hidden_count,
        false,
        self.func
    )


    -- Bias Copying
    for i = 0x01, #self.biases do new.biases[i] = self.biases[i] end

    for i = 0x01, mut_chance*#new.biases do
        local j = math.random(#new.biases)
        new.biases[j] = new.biases[j]+(math.random()*0x02-0x01)*mut_amount
    end


    -- Weight Copying
    for i = 0x01, #self.weights do new.weights[i] = self.weights[i] end

    for i = 0x01, mut_chance*#new.weights do
        local j = math.random(#new.weights)
        new.weights[j] = new.weights[j]+(math.random()*0x02-0x01)*mut_amount
    end


    -- De-initialization
    return new
end




-- Neural Net To String Metafunction
function neural_net.__tostring(self)


    -- Argument Checking
    assert(getmetatable(self) == neural_net, "\"self\" is of wrong type")


    -- Initialization
    local str = ""

    str = str.."in_size      = "..self.in_size      .."\n"
    str = str.."out_size     = "..self.out_size     .."\n"
    str = str.."hidden_size  = "..self.hidden_size  .."\n"
    str = str.."hidden_count = "..self.hidden_count .."\n"
    str = str.."biases       = "..#self.biases      .."\n"
    str = str.."weights      = "..#self.weights     .."\n"


    -- De-initialization
    return str
end




-- De-initialization
return neural_net
