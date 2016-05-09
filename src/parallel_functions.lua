-- Worker code
function worker()
    -- Alert successfully started up
    parallel.print('Im a worker, my ID is: ',  parallel.id, ' and my IP: ', parallel.ip)

    -- Global indicating is a child
    ischild = true
    ext = ""

    -- Number of packages received
    local n_pkg = 0

    while true do
        -- Allow the parent to terminate the child
        m = parallel.yield()
        if m == 'break' then break end   

        -- Receive data
        local pkg = parallel.parent:receive()
        if n_pkg == 0 then 
            -- This is the first time receiving a package, it has the globals

            parallel.print('Recieved initialization parameters')
            cmd, arg, ext = pkg.cmd, pkg.arg, pkg.ext

            -- Load in functions
            print(ext.."model_functions.lua")
            funcs = loadfile(ext .. "model_functions.lua")
            funcs()

            -- Load in data
            datafun = loadfile(ext .. "data.lua")
            data = datafun()
            
            -- Load in data to client
            train_data, valid_data, model, criterion, opt = main()


            --point the wordvec to the right place
            opt.pre_word_vecs = opt.extension .. opt.pre_word_vecs

            first = false

            -- send some data back
            parallel.parent:send('Received parameters and loaded data successfully')
        else
            -- Make sure to clean everything up since big files are being passed
            io.write('.') io.flush()
            collectgarbage()

            parallel.print('received params from batch with index: ', pkg.index)

            -- Load in the parameters sent from the parent
            for i = 1, #model.params do
                model.params[i]:copy(pkg.parameters[i])
            end


            -- Training the model at the given index
            local pkg_o = train_ind(pkg.index, model, criterion, train_data)

            -- send some data back
            parallel.print('sending back derivative for batch with index: ', pkg.index)
            parallel.parent:send(pkg_o)
        end
        n_pkg = n_pkg + 1
    end
end

-- The parent process function
function parent()
    -- Load in the class that runs the server
    require 'sgd_server'

    -- Print from parent process
    parallel.print('Im the parent, my ID is: ',  parallel.id, ' and my IP: ', parallel.ip)

    -- Initialize Server from server.lua class
    param_server = sgd_server.new()

    -- Fork clients and execute startup code
    param_server:fork_and_exec(worker)

    -- Run the server
    param_server:run()   
end