

-- The parent process function
function parent()
    -- Load in the class that runs the server
    require 'sgd_server'

    -- Print from parent process
    parallel.print('Im the parent, my ID is: ',  parallel.id, ' and my IP: ', parallel.ip)

    -- Initialize Server from server.lua class
    param_server = sgd_server.new(opt)

    -- Run the server
    param_server:run()   
end