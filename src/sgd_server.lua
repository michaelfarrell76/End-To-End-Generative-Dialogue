------------------------------------------------------------------------
-- sgd_server.lua
--
-- This is a class that is used to launch the sgd server for this model
--      This class has an _init(opt) function that takes in
--      the global parameters, loads in the data and builds the model on 
--      the parameter server. The class also has a run() function that
--      forks out the child clients and executes the function 'worker'
--      on each corresponding client. 
--
------------------------------------------------------------------------
local sgd_server = torch.class('sgd_server')

------------
-- Worker code
------------
function worker()
    -- Used to check files 
    require "lfs"

    -- Used to update path
    require 'package'
    
    -- Alert successfully started up
    parallel.print('Im a worker, my ID is: ',  parallel.id, ' and my IP: ', parallel.ip)

    -- Global indicating is a child
    ischild = true

    -- Extension to lua-lua folder from home directory. Set to no extension as default
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

        if n_pkg == 0 then 
            -- This is the first time receiving a package, it has the globals

            -- Receive and parse global parameters
            parallel.print('Recieved initialization parameters')
            cmd, arg, ext = pkg.cmd, pkg.arg, pkg.ext
            opt = cmd:parse(arg)
            opt.print = parallel.print

            if opt.gpuid >= 0 then
                print('Using CUDA on GPU ' .. opt.gpuid .. '...')
                if opt.gpuid2 >= 0 then
                    print('Using CUDA on second GPU ' .. opt.gpuid2 .. '...')
                end
                require 'cutorch'
                require 'cunn'
                cutorch.setDevice(opt.gpuid)
                cutorch.manualSeed(opt.seed)
            end

            -- Update path
            package.path = opt.add_to_path .. package.path

            -- Add in additional necessary parameters
            
            opt.parallel = true


             -- Library used to handle data types
            local data_loc = ext .. 'data'
            if not lfs.attributes(data_loc .. '.lua') then
                print('The file data.lua could not be found in ' .. data_loc .. '.lua')
                os.exit()
            end
            data = require(data_loc)

            -- Load in helper functions for this model defined in End-To-End-Generative-Dialogue
            local model_funcs_loc = ext .. "model_functions.lua"
            if not lfs.attributes(model_funcs_loc) then
                print('The file model_functions.lua could not be found in ' .. model_funcs_loc)
                os.exit()
            end
            funcs = loadfile(model_funcs_loc)
            funcs()

            -- Change the locations of the datafiles based on new extension
            opt.data_file = ext .. opt.data_file
            opt.val_data_file = ext .. opt.val_data_file

            --point the wordvec to the right place if exists
            if opt.pre_word_vecs ~= "" then
                opt.pre_word_vecs = opt.extension .. opt.pre_word_vecs
            end
            
            -- Load in data to client
            train_data, valid_data, opt = load_data(opt)

            -- Build the model on the client
            model, criterion = build()

            -- send some data back
            parallel.parent:send('Received parameters and loaded data successfully')
        else

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


------------
-- Server class
------------

-- Initialization function for the server object. Here we load in the data, build our
--      model, and then add any remote client objects if necessary. 
function sgd_server:__init(opt)
    -- Save the command line options 
    self.opt = opt

    -- Used to check files 
    require "lfs"

     -- Library used to handle data types
    local data_loc = 'data'
    if not lfs.attributes(data_loc .. '.lua') then
        print('The file data.lua could not be found in ' .. data_loc .. '.lua')
        os.exit()
    end
    data = require(data_loc)

    -- Load in helper functions for this model defined in End-To-End-Generative-Dialogue
    local model_funcs_loc = "model_functions.lua"
    if not lfs.attributes(model_funcs_loc) then
        print('The file model_functions.lua could not be found in ' .. model_funcs_loc)
        os.exit()
    end
    funcs = loadfile(model_funcs_loc)
    funcs()

    -- Load in the data
    self:load_data()

    -- Setup and build the model
    self:build()

    -- Add remote computers if necessary
    if self.opt.remote then
        parallel.print('Runnings clients remotely')
        
        -- Open the list of client ip addresses
        local fh,err = io.open("../../../client_list.txt")
        if err then print("../../../client_list.txt not found"); return; end

        -- line by line
        while true do
            local line = fh:read()
            if line == nil then break end
            local addr = self.opt.username .. '@' .. line
            addr = string.gsub(addr, "\n", "") -- remove line breaks

            -- Add the remote server by ip address
            parallel.addremote( {ip=addr, cores=4, lua=self.opt.torch_path, protocol='ssh -ttq -o "StrictHostKeyChecking no" -i ~/.ssh/dist-sgd-sshkey'})
            parallel.print('Adding address ', addr)
        end
    elseif opt.localhost then
        -- Has remote clients launched through localhost
        parallel.print('Running clients through localhost')

        parallel.addremote({ip='localhost', cores=4, lua=self.opt.torch_path, protocol='ssh -o "StrictHostKeyChecking no" -i ~/.ssh/dist-sgd-sshkey'})
    end
end

-- Main function that runs the server. Here the child clients are forked off and
--      the code in the 'worker' function is sent to the clients to be run. Once
--      the connection is established, :send() and :recieve() are used to pass 
--      parameters between the client and the server
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

-- Function loads in the training and validation data into self.train_data and
--      seld.valid_data. 
function sgd_server:load_data()
    -- Simply calls the load_data function defined in "End-To-End-Generative-Dialogue/src/model_functions.lua"
    self.train_data, self.valid_data, self.opt = load_data(self.opt)
end

-- Function loads in the nn model and criterion into self.model and self.criterion
function sgd_server:build()
    -- Simply calls the build function defined in "End-To-End-Generative-Dialogue/src/model_functions.lua"
    self.model, self.criterion = build()
end

-- Return the server
return sgd_server
