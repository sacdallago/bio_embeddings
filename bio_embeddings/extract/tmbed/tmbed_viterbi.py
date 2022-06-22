import torch
import torch.nn as nn


class Decoder(nn.Module):

    def __init__(self):
        super().__init__()

        self._init_transitions()

    def _init_transitions(self):
        num_tags = 27

        end_transitions = torch.full((num_tags,), -100)
        start_transitions = torch.full((num_tags,), -100)

        transitions = torch.full((num_tags, num_tags), -100)

        for i in [0, 5, 10, 15, 20, -2, -1]:
            start_transitions[i] = 0  # B1a, B1b, H1a, H1b, S1, i, o

        for i in range(4):
            transitions[0+i, 1+i] = 0    # Bxa -> Bya
            transitions[5+i, 6+i] = 0    # Bxb -> Byb
            transitions[10+i, 11+i] = 0  # Hxa -> Hya
            transitions[15+i, 16+i] = 0  # Hxb -> Hyb
            transitions[20+i, 21+i] = 0  # Sx  -> Sy

        for i in [4, 9, 14, 19, 24]:
            transitions[i, i] = 0  # X5 -> X5

        transitions[4, -1] = 0    # B5a -> o
        transitions[9, -2] = 0    # B5b -> i
        transitions[14, -1] = 0   # H5a -> o
        transitions[19, -2] = 0   # H5b -> i
        transitions[24, -2:] = 0  # S5  -> (i, o)

        transitions[-2, 0] = 0    # i -> B1a
        transitions[-2, 10] = 0   # i -> H1a
        transitions[-2, -2:] = 0  # i -> (i, o)

        transitions[-1, 5] = 0    # o -> B1b
        transitions[-1, 15] = 0   # o -> H1b
        transitions[-1, -2:] = 0  # o -> (i, o)

        for i in [4, 9, 14, 19, 24, -2, -1]:
            end_transitions[i] = 0  # B5a, B5b, H5a, H5b, S5, i, o

        repeats = torch.tensor([10, 10, 5, 1, 1], dtype=torch.int32)

        mapping = torch.arange(7, dtype=torch.int32)
        mapping = mapping.repeat_interleave(torch.tensor([5, 5,  # B
                                                          5, 5,  # H
                                                          5,     # S
                                                          1,     # i
                                                          1]))   # o

        assert repeats.sum() == num_tags
        assert mapping.shape == (num_tags,)

        self.register_buffer('transitions', tensor=transitions)
        self.register_buffer('end_transitions', tensor=end_transitions)
        self.register_buffer('start_transitions', tensor=start_transitions)

        self.register_buffer('repeats', tensor=repeats)
        self.register_buffer('mapping', tensor=mapping)

    def forward(self, emissions, mask):
        mask = mask.transpose(0, 1).bool()

        emissions = emissions.permute(2, 0, 1)
        emissions = emissions.repeat_interleave(self.repeats, dim=2)

        decoded = self._viterbi_decode(emissions, mask)
        decoded = self.mapping[decoded]

        return decoded

    def _viterbi_decode(self, emissions, mask):
        device = emissions.device

        seq_length, batch_size, num_tags = emissions.shape

        score = self.start_transitions + emissions[0]

        history = torch.zeros((seq_length, batch_size, num_tags),
                              dtype=torch.long, device=device)

        for i in range(1, seq_length):
            next_score = (self.transitions
                          + score.unsqueeze(2)
                          + emissions[i].unsqueeze(1))

            next_score, indices = next_score.max(dim=1)

            score = torch.where(mask[i].unsqueeze(-1), next_score, score)

            history[i - 1] = indices

        score = score + self.end_transitions

        _, end_tag = score.max(dim=1)

        seq_ends = mask.long().sum(dim=0) - 1

        history = history.transpose(1, 0)

        history.scatter_(1,
                         seq_ends.view(-1, 1, 1).expand(-1, 1, num_tags),
                         end_tag.view(-1, 1, 1).expand(-1, 1, num_tags))

        history = history.transpose(1, 0)

        best_tags = torch.zeros((batch_size, 1), dtype=torch.long,
                                device=device)

        best_tags_arr = torch.zeros((seq_length, batch_size), dtype=torch.long,
                                    device=device)

        for idx in range(seq_length - 1, -1, -1):
            best_tags = torch.gather(history[idx], 1, best_tags)

            best_tags_arr[idx] = best_tags.view(batch_size)

        return best_tags_arr.transpose(0, 1)
