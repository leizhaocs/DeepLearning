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
Transition::Transition(int n, int h, int w, int final_state)
{
    reward_ = 0;
    action_ = new Action();
    current_ = new Observation(n, h, w);
    next_ = new Observation(n, h, w);

    final_state_ = final_state;
}

/* destructor */
Transition::~Transition()
{
    delete action_;
    delete current_;
    delete next_;
}

/* get r_t+1 */
float &Transition::getReward()
{
    return reward_;
}

/* get a_t */
Action *Transition::getAction()
{
    return action_;
}

/* get s_t */
Observation *Transition::getCurrent()
{
    return current_;
}

/* get s_t+1 */
Observation *Transition::getNext()
{
    return next_;
}

/* to see if next_ is the final state of an episode */
int &Transition::get_final_state()
{
    return final_state_;
}
