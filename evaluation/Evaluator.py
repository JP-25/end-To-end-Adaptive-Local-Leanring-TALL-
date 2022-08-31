import numpy as np

from evaluation.backend import HoldoutEvaluator, predict_topk
from dataloader.DataBatcher import DataBatcher


class Evaluator:
    def __init__(self, eval_pos, eval_target, eval_neg_candidates, ks, num_users=None, num_items=None, item_id=None):
        self.top_k = ks if isinstance(ks, list) else [ks]
        self.max_k = max(self.top_k)

        self.batch_size = 1024
        self.eval_pos = eval_pos
        self.eval_target = eval_target
        self.eval_neg_candidates = eval_neg_candidates
        self.item_id = item_id

        if num_users == None or num_items == None:
            self.num_users, self.num_items = eval_pos.shape
        else:
            self.num_users, self.num_items = num_users, num_items

        ks = sorted(ks) if isinstance(ks, list) else [ks]
        self.eval_runner = HoldoutEvaluator(ks, self.eval_pos, self.eval_target, self.eval_neg_candidates)

    def evaluate(self, model, mean=True):
        # Switch to eval mode
        model.eval()

        # eval users
        eval_users = list(self.eval_target.keys())
        user_iterator = DataBatcher(eval_users, batch_size=self.batch_size)

        score_cumulator = None

        for batch_user_ids in user_iterator:
            # need refactoring
            batch_eval_target = {u: self.eval_target[u] for u in batch_user_ids}
            #   make prediction
            batch_pred = model.predict(batch_user_ids, self.eval_pos)

            # compute metrics
            batch_topk = predict_topk(batch_pred.astype(np.float32), self.max_k).astype(np.int64)
            # print("L43: ", batch_topk)
            # print("L44: ", batch_eval_target)
            score_cumulator = self.eval_runner.compute_metrics(batch_topk, batch_eval_target, score_cumulator)

            # doing it on item side
            ## add here exposure@k
        #     num_interactions = np.sum(self.eval_pos.toarray(), axis=0)
        #     # print("L51: ", num_interactions)
        #     # recommendation_prob = []
        #     exposure_k = {}
        #
        #     for key in self.item_id:
        #         num_exposure = 0
        #         for idx, u in enumerate(batch_eval_target):
        #             pred_u = batch_topk[idx]
        #             target_u = batch_eval_target[u]
        #             num_target_items = len(target_u)
        #             for k in self.top_k:
        #                 # exposure = []
        #                 if k not in exposure_k.keys():
        #                     exposure_k[k] = []
        #                 pred_k = pred_u[:k]
        #                 for item_pred in pred_k:
        #                     if item_pred == key:
        #                         num_exposure = num_exposure + 1
        #                 if idx == len(batch_eval_target) - 1:
        #                     exposure_k[k].append(num_exposure)
        #
        #             # exposure.append(exposure_k)
        #             # print("L65: ", exposure)
        # # get all the item listed recommended probability
        # # print("L76: ", len(exposure_k))
        # # print("L77: ", len(exposure_k[100]))
        # rec_prob = {}
        # for k in self.top_k:
        #     recommendation_prob = exposure_k[k] / (self.num_users - num_interactions)
        #     # print(recommendation_prob) # check here to see if the output is optimal
        #     rec_prob[k] = recommendation_prob
        # # print(rec_prob)
        scores = {}
        for metric in score_cumulator:
            score_by_ks = score_cumulator[metric]
            for k in score_by_ks:
                if mean:
                    scores['%s@%d' % (metric, k)] = score_by_ks[k].mean
                else:
                    scores['%s@%d' % (metric, k)] = score_by_ks[k].history

        # return
        model.train()
        return scores

    def evaluate_partial(self, model, candidate_users=None, mean=True):
        if candidate_users is None:
            print('Candidate users are not privided. Evaluate on all users')
            return self.evaluate(model)

        # Switch to eval mode
        model.eval()

        # eval users
        eval_users = candidate_users
        user_iterator = DataBatcher(eval_users, batch_size=self.batch_size)

        score_cumulator = None

        for batch_user_ids in user_iterator:
            batch_eval_target = {u: self.eval_target[u] for u in batch_user_ids}
            #   make prediction
            batch_pred = model.predict(batch_user_ids, self.eval_pos)

            # compute metrics
            batch_topk = predict_topk(batch_pred.astype(np.float32), self.max_k).astype(np.int64)
            score_cumulator = self.eval_runner.compute_metrics(batch_topk, batch_eval_target, score_cumulator)

        scores = {}
        for metric in score_cumulator:
            score_by_ks = score_cumulator[metric]
            for k in score_by_ks:
                if mean:
                    scores['%s@%d' % (metric, k)] = score_by_ks[k].mean
                else:
                    scores['%s@%d' % (metric, k)] = score_by_ks[k].history

        # return
        model.train()
        return scores

    def update(self, eval_pos=None, eval_target=None, eval_neg_candidates=None):
        if eval_pos is not None:
            self.eval_pos = eval_pos
        if eval_target is not None:
            self.eval_target = eval_target
        if eval_neg_candidates is not None:
            self.eval_neg_candidates = eval_neg_candidates
