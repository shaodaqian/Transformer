
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from model.transformer import Transformer, subsequent_mask
from model.transformer import src_mask as pad_mask
from nltk.translate.bleu_score import sentence_bleu
import copy
import numpy as np


class TranslatorParallel(nn.DataParallel):
    def translate_sentence(self, *args):
        return self.module.translate_sentence(*args)


class Translator(nn.Module):
    def __init__(
            self, model, beam_size, max_seq_len,
            src_pad_idx, trg_pad_idx, trg_bos_idx, trg_eos_idx,device, alpha=0.6):

        super(Translator, self).__init__()

        self.alpha = alpha
        print(self.alpha)
        self.beam_size = beam_size
        self.max_seq_len = max_seq_len
        self.src_pad_idx = src_pad_idx
        self.trg_bos_idx = trg_bos_idx
        self.trg_eos_idx = trg_eos_idx

        self.model = model
        self.model.eval()
        self.device=device
        # self.init_seq=torch.ones(1, 1,device=device,dtype=torch.long).fill_(trg_bos_idx)
        # self.blank_seqs=torch.full((beam_size, max_seq_len), trg_pad_idx, dtype=torch.long,device=device)
        # self.blank_seqs[:, 0] = self.trg_bos_idx
        # self.lenthmap=torch.arange(1, max_seq_len + 1, dtype=torch.long,device=device).unsqueeze(0)

        self.register_buffer('init_seq', torch.LongTensor([[trg_bos_idx]]))
        self.register_buffer(
            'blank_seqs',
            torch.full((beam_size, 1), trg_pad_idx, dtype=torch.long))
        self.blank_seqs[:, 0] = self.trg_bos_idx
        # self.register_buffer(
        #     'len_map',
        #     torch.arange(1, max_seq_len + 1, dtype=torch.long).unsqueeze(0))

    def _model_decode(self, trg_seq, enc_output, src_mask):
        trg_mask = subsequent_mask(trg_seq)
        dec_output = self.model.decode(trg_seq, enc_output, src_mask, trg_mask)
        output = self.model.generator(dec_output[:, -1])
        return output

    def _get_init_state(self, src_seq, src_mask):
        beam_size = self.beam_size
        enc_output = self.model.encode(src_seq, src_mask)
        dec_output = self._model_decode(self.init_seq, enc_output, src_mask)
        # print(dec_output.shape)
        # print(torch.argmax(dec_output[0][0]))
        # print(torch.max(dec_output[0][0]))
        # best_k_probs, best_k_idx = torch.topk(dec_output,1, dim=1)
        best_k_probs, best_k_idx = torch.topk(dec_output, beam_size, dim=1)

        scores = best_k_probs[0]
        # print(scores)
        gen_seq = copy.deepcopy(self.blank_seqs)
        gen_seq = torch.cat([gen_seq, best_k_idx[0].unsqueeze(1)], dim=1)
        enc_output = enc_output.repeat(beam_size, 1, 1)
        # print(gen_seq)

        return enc_output, gen_seq, scores

    def _get_the_best_score_and_idx(self, gen_seq, dec_output, scores, step):
        assert len(scores.size()) == 1
        beam_size = self.beam_size

        # Get k candidates for each beam, k^2 candidates in total.
        # torch.topk(dec_output[:, -1, :], beam_size)
        best_k2_probs, best_k2_idx = torch.topk(dec_output, beam_size, dim=1)
        # Include the previous scores.
        scores = best_k2_probs.view(beam_size, -1) + scores.view(beam_size, 1)
        # Get the best k candidates from k^2 candidates.
        scores, best_k_idx_in_k2 = scores.view(-1).topk(beam_size)
        # print(best_k_idx_in_k2)

        # print(best_k_idx_in_k2,'best_k_idx_in_k2')
        # Get the corresponding positions of the best k candidiates.
        best_k_r_idxs = best_k_idx_in_k2 // beam_size
        best_k_c_idxs = best_k_idx_in_k2 % beam_size
        # print(gen_seq.shape)
        # print(best_k_r_idxs,'best_k_r_idxs')
        # Copy the corresponding previous tokens.
        # temp=copy.deepcopy(gen_seq)
        gen_seq[:, :step] = gen_seq[best_k_r_idxs, :step]
        # Set the best tokens in this beam search step
        # print(best_k2_idx[best_k_r_idxs, best_k_c_idxs])
        gen_seq = torch.cat([gen_seq, best_k2_idx[best_k_r_idxs, best_k_c_idxs].unsqueeze(1)], dim=1)
        return gen_seq, scores

    def translate_sentence(self, source_sequence):
        src_pad_idx, trg_eos_idx = self.src_pad_idx, self.trg_eos_idx
        max_seq_len, beam_size, alpha = self.max_seq_len, self.beam_size, self.alpha
        ans=[]
        ends=[]
        for i in range(source_sequence.size(0)):
            with torch.no_grad():
                src_seq=source_sequence[i].unsqueeze(0)
                src_mask = pad_mask(src_seq, src_pad_idx)
                enc_output, gen_seq, scores = self._get_init_state(src_seq, src_mask)
                ans_idx = 0  # default
                for step in range(2, max_seq_len):  # decode up to max length
                    dec_output = self._model_decode(gen_seq[:, :step], enc_output, src_mask)
                    gen_seq, scores = self._get_the_best_score_and_idx(gen_seq, dec_output, scores, step)
                    # Check if all path finished
                    # -- locate the eos in the generated sequences
                    eos_locs = (gen_seq == trg_eos_idx)
                    eos_locs=eos_locs.to(self.device)
                    # eos_locs=eos_locs.type(torch.IntTensor)
                    # -- replace the eos with its position for the length penalty use
                    # print(self.len_map)
                    # seq_lens,_ = self.lenthmap.masked_fill(~eos_locs, max_seq_len).min(1)
                    # -- check if all beams contain eos
                    # print(eos_locs.sum())
                    end=step
                    if (eos_locs.sum(1) > 0).sum(0).item() == beam_size:
                        # print(gen_seq)
                        seq_len=torch.argmax(eos_locs.int().to(self.device),dim=1)
                        _, ans_idx = scores.div(((seq_len.float().to(self.device)+5)** alpha).div(6**alpha)).max(0)
                        ans_idx = ans_idx.item()
                        end=torch.argmax(eos_locs[ans_idx].int().to(self.device))
                        break
            # print(gen_seq[ans_idx])
            # print(end)
            ans.append(gen_seq[ans_idx].tolist())
            ends.append(end)
        return torch.tensor(ans),ends

    def greedy_decoder(self, source_sequence):
        src_pad_idx, trg_eos_idx, trg_bos_idx = self.src_pad_idx, self.trg_eos_idx, self.trg_bos_idx
        max_seq_len, beam_size, alpha = self.max_seq_len, self.beam_size, self.alpha
        src_mask = pad_mask(source_sequence, src_pad_idx)
        memory = self.model.encode(source_sequence, src_mask)
        ys = torch.full((1, 1),trg_bos_idx,dtype=torch.long)
        for i in range (max_seq_len-1):
            out = self.model.decode(ys,memory, src_mask, subsequent_mask(ys))
            prob = self.model.generator(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.data[0]
            ys = torch.cat([ys,torch.ones(1, 1).type(torch.LongTensor).fill_(next_word)], dim=1)
        print(ys)
        return ys