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

#ifndef _TRANSITION_H_
#define _TRANSITION_H_

#include "includes.h"

/* a transition, a tuple of (s_t, a_t, r_t+1, s_t+1) */
class Transition
{
public:
    /* constructor */
    Transition(int n, int h, int w, int final_state);

    /* destructor */
    ~Transition();

    /* get r_t+1 */
    float &getReward();

    /* get a_t */
    Action *getAction();

    /* get s_t */
    Observation *getCurrent();

    /* get s_t+1 */
    Observation *getNext();

    /* to see if next_ is the final state of an episode */
    int &get_final_state();
    
private:
    float reward_;          // r_t+1
    Action *action_;        // a_t
    Observation *current_;  // s_t
    Observation *next_;     // s_t+1

    int final_state_;       // 1 when next_ is the final state of an episode, otherwise 0
};

#endif
