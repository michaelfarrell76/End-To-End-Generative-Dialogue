require 'rnn'

local INF = 1e9
local beam = torch.class('beam')

------------
-- Misc
------------

-- Convert a flat index to a matrix
local function flat_to_rc(v, indices, flat_index)
    local row = math.floor((flat_index - 1) / v:size(2)) + 1
    return row, indices[row][(flat_index - 1) % v:size(2) + 1]
end

-- Find the k max in a vector, but in a janky way cause fbcunn requires linux
local function find_k_max(mat, K)
    local scores = torch.Tensor(mat:size(1), K)
    local indices = torch.Tensor(mat:size(1), K)
    for i = 1, mat:size(1) do
        for k = 1, K do
            local score, index = mat[i]:max(1)
            score = score[1]; index = index[1]
            scores[i][k] = score
            indices[i][k] = index
            mat[i][index] = mat[i][index] - INF
        end
    end
    return scores, indices
end

------------
-- Scoring
------------

local function forward_connect(enc_rnn, dec_rnn, seq_length, layer_type)
	if layer_type == 'bi' then
        enc_fwd_seqLSTM = enc_rnn['modules'][1]['modules'][2]['modules'][1]
        enc_bwd_seqLSTM = enc_rnn['modules'][1]['modules'][2]['modules'][2]['modules'][2]
        
        dec_rnn.userPrevOutput = enc_rnn.output[{{}, seq_length}]
        dec_rnn.userPrevCell = enc_bwd_seqLSTM.cell[seq_length]
    else
        dec_rnn.userPrevOutput = nn.rnn.recursiveCopy(dec_rnn.userPrevOutput,
        	enc_rnn.outputs[seq_length])
        if layer_type ~= 'gru' and layer_type ~= 'rnn' then
            dec_rnn.userPrevCell = nn.rnn.recursiveCopy(dec_rnn.userPrevCell,
            	enc_rnn.cells[seq_length])
        end
    end
end

-- Artificially limits specific vocabulary
local function constrain(scores, allow_unk)
	-- scores[{{}, START}] = -INF
	scores[{{}, 8}] = -INF -- Disallow <person>
	if allow_unk == 0 then
	    scores[{{}, UNK}] = -INF
	end
	return scores
end

-- Returns encoder decoder scores accounting for context (standard forward pass)
local function get_scores(m, source, cur_beam)
    local source_l = source:size(1)
    source = source:contiguous()
    source = source:view(1, -1):expand(cur_beam:size(1), source_l)

    -- Forward prop enc + dec
    local enc_out = m.enc:forward(source)
    forward_connect(m.enc_rnn, m.dec_rnn, source_l)
    local preds = m.dec:forward(cur_beam)

    -- Return log probability distribution for all words
    -- NB: preds[#preds] = log prob of most recent word
    return preds
end

-- Returns a pure language model score without any preceding context
local function get_lm_scores(lm, cur_beam)
	return lm:forward(cur_beam)
end


-- Prevent sentences without content or only punctuation
local function validate(hyp)
	return hyp[2] ~= END and hyp[2] ~= 10003 and hyp[2] ~= 9
end

------------
-- Beam class
------------

function beam:__init(opt, model, lm)
    self.opt = opt
    self.m = model
    self.m.enc:evaluate()
    self.m.dec:evaluate()
    self.lm = lm
end

-- Generates K most likely output utterances given an input source
function beam:generate(K, source, gold)
    -- Let's get all fb up in here
    -- scores[i][k] is the log prob of the k'th hyp of i words
    -- hyps[i][k] contains the words in k'th hyp at i word
    local n = self.opt.max_sent_l or 20
    local full = false
    local result = {}
    local scores = torch.zeros(n + 1, K):float()
    local hyps = torch.zeros(n + 1, K, n + 1):long()
    hyps:fill(START)


    -- Beam me up, Scotty!
    for i = 1, n do
        if full then break end

        local cur_beam = hyps[i]:narrow(2, 1, i)
        -- print(cur_beam)
        local cur_K = K

        -- Score all next words for each context in the beam
        -- log p(y_{i+1} | y_c, x) for all y_c
        local all = get_scores(self.m, source, cur_beam)

        local out = all[#all]

        -- Only take first row when starting out as all predictions are same
        if i == 1 then
            cur_K = 1
            out = out:narrow(1, 1, 1)
        end

        -- Apply hard constraints to certain vocabulary
    	out = constrain(out, self.opt.allow_unk)

        -- Recaculate scores for each sequence using MMI-antiLM
        if self.opt.antilm == 1 and i <= self.opt.gamma then
        	local lm = get_lm_scores(self.lm, cur_beam)
            lm = lm[#lm]

            for k = 1, cur_K do
                out[k]:csub(lm[k] * self.opt.lambda)
            end
        end

        -- Prob of hypothesis is log prob of preceding sequence + log prob of
        -- most recent predictions
        for k = 1, cur_K do
            out[k]:add(scores[i][k])
        end

        -- Keep only the K-best words for each hypothesis
        -- This leaves a KxK matrix which we flatten to a K^2 vector
        local max_scores, mat_indices = find_k_max(out, K)
        local flat = max_scores:view(max_scores:size(1) *
            max_scores:size(2)):float()

        -- Construct the next hypotheses by taking the next k-best
        for k = 1, K do
            -- Pull the score, index, rank, and word of the current best
            -- in the table, and then zero it out
            local score, index = flat:max(1)
            scores[i+1][k] = score[1]

            local prev_k, y_i1 = flat_to_rc(max_scores, mat_indices, index[1])
            flat[index[1]] = -INF

            -- Add the word and its score to the beam
            -- Update tables with new hypothesis
            for j = 1, i do
                local pword = hyps[i][prev_k][j]
                hyps[i+1][k][j] = pword
            end
            hyps[i+1][k][i+1] = y_i1

            -- If we have produced an END symbol and valid utterance, push to stack
            if y_i1 == END and validate(hyps[i+1][k]) then
            	-- Normalize probability over length
            	-- Not *that* helpful, but right idea
            	-- local norm_score = scores[i+1][k] / (i + 1)
                -- table.insert(result, {i+1, scores[i+1][k] +  i * self.opt.len_reward, hyps[i+1][k]:clone()})
                table.insert(result, {i+1, scores[i+1][k] + torch.log(i) * 3, hyps[i+1][k]:clone()})
                scores[i+1][k] = -INF

                -- if #result == K then
                --     full = true
                --     break
                -- end
            end
        end
    end

    -- Sort by score, and we're done
    table.sort(result, function (a, b) return a[2] > b[2] end)
    return result
end

-- Generates most likely output utterance (map) given an input source
function beam:generate_map(source, gold)
    local result = self:generate(self.opt.k or 5, source, gold)
    local len = result[1][1]
    local score = result[1][2]
    local sent = result[1][3]
    return sent[{{1, len}}], score
end

-- For external use
-- function beam:generate_k(k, source)
--     local result = self:generate(k, source, nil)
--     local outputs = {}
--     local scores = {}
--     for i = 1, #result do
--         -- result[i] = length, score, sentence
--         local len = result[i][1]
--         local score = result[i][2]
--         local sent = result[i][3]
--         sent = sent[{{1, len}}]
--         table.insert(outputs, sent)
--         table.insert(scores, score)
--     end
--     return outputs, scores
-- end

function beam:generate_k(k, source)
    local result = self:generate(k, source, nil)
    local outputs = {}
    local scores = {}
    for i = 1, self.opt.k2 do
        -- result[i] = length, score, sentence
        local len = result[i][1]
        local score = result[i][2]
        local sent = result[i][3]
        sent = sent[{{1, len}}]
        table.insert(outputs, sent)
        table.insert(scores, score)
    end
    return outputs, scores
end
return beam
