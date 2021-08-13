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

#ifndef _PARAMS_H_
#define _PARAMS_H_

#include "includes.h"

/* arguments pass to the network */
class Params
{
private:
    /* type */
    enum elem_type
    {
        STRING,   // string
        SCALARI,  // integer scalar
        SCALARF,  // floating point scalar
        VECTORI,  // integer vector
        VECTORF   // floating point vector
    };

    /* element of one field */
    struct element
    {
        string name;               // name of this field
        elem_type type;            // type of the data

        string str;                // string
        int scalri;                // integer scalar
        float scalrf;              // floating point scalar
        vector<int> vectri;        // integer vector
        vector<float> vectrf;      // floating point vector
    };

    vector<struct element> elems_;  // all the elements

public:
    /* add a string field */
    void addString(const char *fieldname, const char *value);

    /* add a scalari field */
    void addScalari(const char *fieldname, int value);

    /* add a scalarf field */
    void addScalarf(const char *fieldname, float value);

    /* add a vectori field */
    void addVectori(const char *fieldname, vector<int> value);

    /* add a vectorf field */
    void addVectorf(const char *fieldname, vector<float> value);

    /* check if a field exists */
    bool hasField(const char *fieldname);

    /* get a specific string field */
    string getString(const char *fieldname);

    /* get a specific scalari field */
    int getScalari(const char *fieldname);

    /* get a specific scalarf field */
    float getScalarf(const char *fieldname);

    /* get a specific vectori field */
    vector<int> getVectori(const char *fieldname);

    /* get a specific vectorf field */
    vector<float> getVectorf(const char *fieldname);

    /* get number of values in a specific field */
    int getNumel(const char *fieldname);
};

#endif
