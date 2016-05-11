require 'dict'

--
-- Manages encoder/decoder data matrices
--

------------
-- HRED
------------

-- Finds indices for all locations of a specific value
function find_all(t, val)
  	local indices = {}
  	for i = 1, t:size(1) do
    	if t[i] == val then
      		table.insert(indices, i)
    	end
  	end
  	return indices
end

function split_utterances(source)
	-- First determine context length
    local std_len = source:size(2)
	local source_split = {}
	for i = 1, opt.utter_context do
		table.insert(source_split, torch.zeros(source:size()))
	end

	-- Split each utterance out into entries in table
	for i = 1, source:size(1) do
		local cleaned = remove_pad(source[i])
		local end_utterances = find_all(cleaned, END_UTTERANCE)

		-- For now it's assumed opt.utter_context = 2
		-- TODO: generalize to opt.utter_context = n
        local first = cleaned:sub(1, end_utterances[1])
        if end_utterances[1] + 1 > cleaned:size(1) then
            -- Necessary to handle a few bad cases where second utterance length = 0
            end_utterances[1] = cleaned:size(1) - 1
        end
        local second = cleaned:sub(end_utterances[1] + 1, cleaned:size(1))

        -- End of first is always going to be end utterance token, which we
        -- want to replace
        first[first:size(1)] = END
        first = pad_start(first)
        second = pad_both(second)

        -- Standardize length by prepending pad tokens
        -- TODO: instead of assuming std length is the length of combined tokens,
        -- use the max seq length found in either utterance in batch (should
        -- result in less padding)
        first = pad_blank(first, std_len)
        second = pad_blank(second, std_len)

        source_split[1][{i, {}}] = first
        source_split[2][{i, {}}] = second
	end

	return source_split
end

------------
-- Data
------------

local data = torch.class('data')

function data:__init(opt, data_file)
   	local f = hdf5.open(data_file, 'r')

   	self.source  = f:read('source'):all()
   	self.target  = f:read('target'):all()
   	self.target_output = f:read('target_output'):all()
   	self.target_l = f:read('target_l'):all() -- Max target length each batch
   	self.target_l_all = f:read('target_l_all'):all()
   	self.target_l_all:add(-1)
   	self.batch_l = f:read('batch_l'):all()
   	self.source_l = f:read('batch_w'):all() -- Max source length each batch
   	if opt.start_symbol == 0 then
      	self.source_l:add(-2)
      	self.source = self.source[{{},{2, self.source:size(2)-1}}]
   	end
   	self.batch_idx = f:read('batch_idx'):all()

   	self.target_size = f:read('target_size'):all()[1]
   	self.source_size = f:read('source_size'):all()[1]
   	self.target_nonzeros = f:read('target_nonzeros'):all()
   
   	self.length = self.batch_l:size(1)
   	self.seq_length = self.target:size(2)
   	self.batches = {}
   	local max_source_l = self.source_l:max()
   	local source_l_rev = torch.ones(max_source_l):long()
   	for i = 1, max_source_l do
      	source_l_rev[i] = max_source_l - i + 1
   	end
   	for i = 1, self.length do
      	local target_output_i = self.target_output:sub(self.batch_idx[i],self.batch_idx[i]
      		+self.batch_l[i]-1, 1, self.target_l[i])
      	local target_l_i = self.target_l_all:sub(self.batch_idx[i],
      		self.batch_idx[i]+self.batch_l[i]-1)
 		local source_i = self.source:sub(self.batch_idx[i], self.batch_idx[i]+
 			self.batch_l[i]-1, 1, self.source_l[i]):transpose(1,2):t()

      	if opt.reverse_src == 1 then
	 		source_i = source_i:index(1, source_l_rev[{{max_source_l-self.source_l[i]+1,
	 			max_source_l}}])
      	end

      	-- For HRED, split utterances apart at each end utterance token
      	if opt.model_type == 'hred' then
      		source_i = split_utterances(source_i)
      	end

 		local target_i = self.target:sub(self.batch_idx[i], self.batch_idx[i]+self.batch_l[i]-1,
 			1, self.target_l[i]):transpose(1,2):t()

      	table.insert(self.batches,  {target_i,
      		target_output_i:transpose(1,2),
			self.target_nonzeros[i],
			source_i,
			self.batch_l[i],
			self.target_l[i],
			self.source_l[i],
			target_l_i})
   	end
end

function data:size()
   	return self.length
end

function data.__index(self, idx)
   	if type(idx) == 'string' then
      	return data[idx]
   	else
      	local target_input = self.batches[idx][1]
      	local target_output = self.batches[idx][2]
      	local nonzeros = self.batches[idx][3]
      	local source_input = self.batches[idx][4]      
      	local batch_l = self.batches[idx][5]
      	local target_l = self.batches[idx][6]
      	local source_l = self.batches[idx][7]
      	local target_l_all = self.batches[idx][8]
      	if opt.gpuid >= 0 then -- If multi-gpu, source lives in gpuid1, rest on gpuid2
	 		cutorch.setDevice(opt.gpuid)
	 		source_input = source_input:cuda()
	 		if opt.gpuid2 >= 0 then
	    		cutorch.setDevice(opt.gpuid2)
	 		end
	 		target_input = target_input:cuda()
	 		target_output = target_output:cuda()
	 		target_l_all = target_l_all:cuda()
      	end
      	return {target_input, target_output, nonzeros, source_input,
	      batch_l, target_l, source_l, target_l_all}
   	end
end

return data
