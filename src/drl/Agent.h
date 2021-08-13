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

#ifndef _AGENT_H_
#define _AGENT_H_

#include "includes.h"

/* an agent */
class Agent
{
public:
    /* constructor */
    Agent(float epsilonStart, float epsilonEnd, float epsilonDecay, float lambda, int numActions,
          Params *layers, int num_layers, int batchSize, string loss);

    /* destructor */
    ~Agent();

    /* select an action based on epsilon-greedy policy */
    int selectAction(Observation *ob);

    /* update policy network*/
    void update_policy(int sampleSize, Transition **samples, float lr);

    /* update target network */
    void update_target();

private:
    float epsilonStart_;        // epsilon-greedy policy
    float epsilonEnd_;          // epsilon-greedy policy
    float epsilonDecay_;        // epsilon-greedy policy
    float lambda_;              // discount rate
    int numActions_;            // number of actions
    int steps_;                 // number of steps this agent has made

    Net *policyNet_;            // policy network
    Net *targetNet_;            // target nerwork
    float *weights_;            // buffer for copying weights from policy network to target nerwork
};

#endif
