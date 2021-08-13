/**********************************************************************
 *
 * Copyright Lei Zhao.
 * contact: leizhao0403@gmail.com
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 **********************************************************************/

#include "includes.h"

/* constructor */
Agent::Agent(float epsilonStart, float epsilonEnd, float epsilonDecay, float lambda, int numActions,
             Params *layers, int num_layers, int batchSize, string loss)
{
    epsilonStart_ = epsilonStart;
    epsilonEnd_ = epsilonEnd;
    epsilonDecay_ = epsilonDecay;
    lambda_ = lambda;
    numActions_ = numActions;
    steps_ = 0;

    policyNet_ = new Net(layers, num_layers, batchSize, loss);
    targetNet_ = new Net(layers, num_layers, batchSize, loss);

    int num_weights = policyNet_->getNumWeights(NULL);
    weights_ = new float[num_weights];

    random_weights(weights_, num_weights, 0, 1, 0.001);
    policyNet_->initWeights(weights_);
    targetNet_->initWeights(weights_);
}

/* destructor */
Agent::~Agent()
{
    delete policyNet_;
    delete targetNet_;
    delete [] weights_;
}

/* select an action based on epsilon-greedy policy */
int Agent::selectAction(Observation *ob)
{
    float epsilon = epsilonEnd_ + (epsilonStart_-epsilonEnd_) * exp(-1. * steps_ * epsilonDecay_);
    steps_++;
    if (rand()/float(RAND_MAX) < epsilon)
    {
        int a = rand() % numActions_;
printf("--------------%d  %f\n", a, epsilon);
        return a;//rand() % numActions_;
    }

    Tensor<float> *pred = new Tensor<float>(1, 1, 1, numActions_);

    policyNet_->initForward(ob->getImagesPtr(), 0, 0, 1, ob->getLength(), ob->getHeight(), ob->getWidth());
    policyNet_->forward(1, false);
    policyNet_->getOutputs(pred, 0, 0, 1, numActions_);

#if GPU == 1
    if (use_gpu)
    {
        pred->toCpu();
    }
#endif

    int action = -1;
    float max = -FLT_MAX;
    for (int i = 0; i < numActions_; i++)
    {
printf("%.1f  ", pred->data(i));
        if (pred->data(i) > max)
        {
            max = pred->data(i);
            action = i;
        }
    }
printf("%d\n", action);

    delete pred;

    return action;
}

/* update policy network */
void Agent::update_policy(int sampleSize, Transition **samples, float lr)
{
    Tensor<float> *policy_pred = new Tensor<float>(sampleSize, 1, 1, numActions_);
    Tensor<float> *target_pred = new Tensor<float>(sampleSize, 1, 1, numActions_);
    Tensor<float> *rewards = new Tensor<float>(1, 1, 1, sampleSize);
    Tensor<int> *actions = new Tensor<int>(1, 1, 1, sampleSize);
    Tensor<int> *final_states = new Tensor<int>(1, 1, 1, sampleSize);

    int c = samples[0]->getCurrent()->getLength();
    int h = samples[0]->getCurrent()->getHeight();
    int w = samples[0]->getCurrent()->getWidth();

    for (int i = 0; i < sampleSize; i++)
    {
        policyNet_->initForward(samples[i]->getCurrent()->getImagesPtr(), 0, i, 1, c, h, w);
        targetNet_->initForward(samples[i]->getNext()->getImagesPtr(), 0, i, 1, c, h, w);
        rewards->data(i) = samples[i]->getReward();
        actions->data(i) = samples[i]->getAction()->getAction();
        final_states->data(i) = samples[i]->get_final_state();
    }

    policyNet_->forward(sampleSize, true);
    targetNet_->forward(sampleSize, false);

    policyNet_->getOutputs(policy_pred, 0, 0, sampleSize, numActions_);
    targetNet_->getOutputs(target_pred, 0, 0, sampleSize, numActions_);

#if GPU == 1
    if (use_gpu)
    {
        rewards->toGpu();
        actions->toGpu();
        final_states->toGpu();
        dqn_target_gpu(target_pred->getGpuPtr(), policy_pred->getGpuPtr(), lambda_, rewards->getGpuPtr(), actions->getGpuPtr(), final_states->getGpuPtr(), sampleSize, numActions_);
    }
    else
    {
        dqn_target(target_pred->getCpuPtr(), policy_pred->getCpuPtr(), lambda_, rewards->getCpuPtr(), actions->getCpuPtr(), final_states->getCpuPtr(), sampleSize, numActions_);
    }
#else
    dqn_target(target_pred->getCpuPtr(), policy_pred->getCpuPtr(), lambda_, rewards->getCpuPtr(), actions->getCpuPtr(), final_states->getCpuPtr(), sampleSize, numActions_);
#endif

    policyNet_->loss(target_pred, 0, sampleSize, numActions_);
    policyNet_->backward(sampleSize);
    policyNet_->update(sampleSize, lr);

    delete final_states;
    delete actions;
    delete rewards;
    delete policy_pred;
    delete target_pred;
}

/* update target network */
void Agent::update_target()
{
    policyNet_->getWeights(weights_);
    targetNet_->initWeights(weights_);
}
