------------------------------------------------------------------------
-- demo_server.lua
------------------------------------------------------------------------
local sgd_server = torch.class('sgd_server')

------------
-- Server class
------------
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
        -- Make sure to clean everything up since big files are being passed
        io.write('.') io.flush()
        collectgarbage()


        print('recevived')
        parallel.print('recevived')
        if n_pkg == 0 then 
            -- This is the first time receiving a package, it has the globals

            parallel.print('Recieved initialization parameters')
            cmd, arg, ext = pkg.cmd, pkg.arg, pkg.ext

            opt = cmd:parse(arg)
            opt.print = parallel.print

            -- Load in functions
            funcs = loadfile(ext .. "model_functions.lua")
            funcs()

            -- Load in data
            datafun = loadfile(ext .. "data.lua")
            data = datafun()

            opt.data_file = ext .. opt.data_file
            opt.val_data_file = ext .. opt.val_data_file
            
            -- Load in data to client
            train_data, valid_data = load_data(opt)
            model, criterion = build()


            --point the wordvec to the right place
            opt.pre_word_vecs = opt.extension .. opt.pre_word_vecs

            parallel.print('a')
            -- send some data back
            parallel.parent:send('Received parameters and loaded data successfully')
        else
            print('b')
            opt.print('b')
            

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

function sgd_server:__init(opt)
    self.opt = opt
    -- Load in helper functions for this model
    funcs = loadfile("model_functions.lua")
    funcs()

    self:load_data()

    self:build()

    if self.opt.remote then
        parallel.print('Runnings clients remotely')
        
        -- Open the list of client ip addresses
        local fh,err = io.open("../client_list.txt")
        if err then print("../client_list.txt not found"); return; end

        -- line by line
        while true do
            local line = fh:read()
            if line == nil then break end
            local addr = self.opt.username .. '@' .. line
            addr = string.gsub(addr, "\n", "") -- remove line breaks
            parallel.addremote( {ip=addr, cores=4, lua=self.opt.torch_path, protocol='ssh -ttq -o "StrictHostKeyChecking no" -i ~/.ssh/gcloud-sshkey'})
            parallel.print('Adding address ', addr)
        end
    elseif opt.localhost then
        parallel.print('Running clients through localhost')

        parallel.addremote({ip='localhost', cores=4, lua=self.opt.torch_path, protocol='ssh -o "StrictHostKeyChecking no" -i ~/.ssh/gcloud-sshkey'})
    end
end

function sgd_server:run()
    parallel.print('Forking ', self.opt.n_proc, ' processes')
    parallel.sfork(self.opt.n_proc)
    parallel.print('Forked')

    -- exec worker code in each process
    parallel.children:exec(worker)
    parallel.print('Finished telling workers to execute')
    --send the global parameters to the children
    parallel.children:join()
    parallel.print('Sending parameters to children')
    parallel.children:send({cmd = cmd, arg = arg, ext = self.opt.extension})

    -- Get the responses from the children
    replies = parallel.children:receive()
    parallel.print('Replies from children', replies)

    -- Train the model
    train(self.model, self.criterion, self.train_data, self.valid_data)
    parallel.print('Finished training the model')

    -- sync/terminate when all workers are done
    parallel.children:join('break')
    parallel.print('All processes terminated')
end

function sgd_server:load_data()
    self.train_data, self.valid_data, self.opt = load_data(self.opt)
end

function sgd_server:build()
    self.model, self.criterion = build()
end


return sgd_server
