-- define code for workers:
function worker()

    funcs = loadfile("functions.lua")
    funcs()

    --load in functions
    datafun = loadfile("data.lua")
    data = datafun()

    -- print from worker:
    parallel.print('Im a worker, my ID is: ' .. parallel.id .. ' and my IP: ' .. parallel.ip)

    first = true

   -- define a storage to receive data from top process
    while true do
        -- yield = allow parent to terminate me
        m = parallel.yield()
        if m == 'break' then break end   

        -- receive data
        local pkg = parallel.parent:receive()
        if first then 
            parallel.print('Recieved initialization parameters')
            cmd = pkg.cmd
            arg = pkg.arg
            
            train_data, valid_data, model, criterion, opt = main()

            first = false

            -- send some data back
            parallel.parent:send('Received parameters and loaded data successfully')
        else
            

            io.write('.') io.flush()
            collectgarbage()

            parallel.print('received object with index: ', pkg.index)


            for i = 1, #model.params do
                model.params[i]:copy(pkg.parameters[i])
            end

            local pkg_o = train_ind(pkg.index, model, criterion, train_data)


            -- send some data back

            parallel.print('sending back object with index: ', pkg.index)
            parallel.parent:send(pkg_o)
        end
    end
end


-- define code for parent:
function parent()
    -- print from top process
    parallel.print('Im the parent, my ID is: ' .. parallel.id)

    train_data, valid_data, model, criterion, opt = main()

    n_proc= 4

    -- fork N processes
    parallel.nfork(n_proc)

    -- parallel.addremote(...)
    -- parallel.calibrate()
    -- forked = parallel.sfork(parallel.remotes.cores)  -- fork as many processes as cores available
    -- for _,forked in ipairs(forked) do
    --    parallel.print('id: ' .. forked.id .. ', speed = ' .. forked.speed)
    -- end

    -- exec worker code in each process
    parallel.children:exec(worker)

    --send the global parameters to the children
    parallel.children:join()
    parallel.children:send({cmd = cmd, arg = arg})
    replies = parallel.children:receive()
    parallel.print(replies)

    --trainmodel
    train(model, criterion, train_data, valid_data)


    parallel.print('transmitted data to all children')

    -- sync/terminate when all workers are done
    parallel.children:join('break')
    parallel.print('all processes terminated')
end