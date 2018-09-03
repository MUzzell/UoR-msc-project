--[[
Copyright (c) 2014 Google Inc.

See LICENSE file for full terms of limited license.
]]

require "nn"
require "image"
require "Scale"

return function(args)
    -- Y (luminance)
    return nn.Scale(84, 84, true)
end

