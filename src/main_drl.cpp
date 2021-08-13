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

/* main function */
int main_drl(int argc, char *argv[])
{
    ReplayMemory *mem;
    Agent *agent;
    Transition **samples;

    /* network parameters */
    Params *layer_params;         // config file parameters
    int num_layers;               // number of layers
    int num_epochs;               // number of epochs
    int batchSize;                // batch size
    float lr_begin;               // learning rate
    float lr_decay;               // decay of learning rate
    string loss;                  // loss function
    int show_acc;                 // show accuracy in each epoch

    /* environment parameters */
    int episodes = 200000;
    int mem_size = 20000;
    int n = 4;
    int c = 0;
    int h = 0;
    int w = 0;
    int done = 0;
    float epsilonStart = 1;
    float epsilonEnd = 0.01;
    float epsilonDecay = 0.001;
    float lambda = 0.999;
    int num_actions = 2;
    int period = 20;

    cout<<"Creating environment..."<<endl;

    if (gym_connect() == -1)
    {
        return -1;
    }
    gym_make("CartPole-v0");
    gym_reset();
    gym_getImageDimension(&c, &h, &w);
    printf("image shape: %d x %d x %d\n", c, h, w);

    cout<<"Preparing structures..."<<endl;

    int image_size = c*h*w*sizeof(float);
    mem = new ReplayMemory(mem_size, n, h, w);

    layer_params = build_network(argv[2], num_layers, num_epochs, batchSize, lr_begin, lr_decay, loss, show_acc);
    agent = new Agent(epsilonStart, epsilonEnd, epsilonDecay, lambda, num_actions, layer_params, num_layers, batchSize, loss);

    samples = new Transition*[batchSize];

    cout<<"Running..."<<endl;

    for (int i = 0; i < episodes; i++)
    {
        gym_reset();
        mem->reset();
        done = 0;

        int steps = 0;
        while (done != 1)
        {
            steps++;

            int action;
            if (steps > 1)
            {
                action = agent->selectAction(mem->getLatestTran()->getNext());
            }
            else
            {
                action = rand() % num_actions;
            }

            float reward = gym_step_discrete(action, &done);
            gym_render(mem->getFrameBuffer(), image_size);
            //float a[] = {0.4, 11.9, 34.9};
            //float reward = gym_step_box(3, a, &done);

            mem->addTransition(reward, action, done);

            bool enough_samples = mem->sampleTransitions(batchSize, samples);
            if (enough_samples)
            {
                agent->update_policy(batchSize, samples, lr_begin);
            }
        }
        if (i % period == 0)
        {
            agent->update_target();
        }
        printf("Episode: %d  steps: %d\n", i, steps);
    }

    gym_close();
    gym_disconnect();

    delete [] layer_params;
    delete mem;
    delete agent;
    delete [] samples;

	return 0;
}
