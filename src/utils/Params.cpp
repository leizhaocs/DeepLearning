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

/* add a string field */
void Params::addString(const char *fieldname, const char *value)
{
    element e;
    string fieldname_str(fieldname);
    e.name = fieldname_str;
    e.type = STRING;
    string value_str(value);
    e.str = value_str;
    elems_.push_back(e);
}

/* add a scalari field */
void Params::addScalari(const char *fieldname, int value)
{
    element e;
    string fieldname_str(fieldname);
    e.name = fieldname_str;
    e.type = SCALARI;
    e.scalri = value;
    elems_.push_back(e);
}

/* add a scalarf field */
void Params::addScalarf(const char *fieldname, float value)
{
    element e;
    string fieldname_str(fieldname);
    e.name = fieldname_str;
    e.type = SCALARF;
    e.scalrf = value;
    elems_.push_back(e);
}

/* add a vectori field */
void Params::addVectori(const char *fieldname, vector<int> value)
{
    element e;
    string fieldname_str(fieldname);
    e.name = fieldname_str;
    e.type = VECTORI;
    e.vectri = value;
    elems_.push_back(e);
}

/* add a vectorf field */
void Params::addVectorf(const char *fieldname, vector<float> value)
{
    element e;
    string fieldname_str(fieldname);
    e.name = fieldname_str;
    e.type = VECTORF;
    e.vectrf = value;
    elems_.push_back(e);
}

/* check if a field exists */
bool Params::hasField(const char *fieldname)
{
    string fieldname_str(fieldname);
    for (int i = 0; i < elems_.size(); i++)
    {
        if (elems_[i].name.compare(fieldname_str) == 0)
        {
            return true;
        }
    }
    return false;
}

/* get a specific string field */
string Params::getString(const char *fieldname)
{
    string fieldname_str(fieldname);
    for (int i = 0; i < elems_.size(); i++)
    {
        if (elems_[i].name.compare(fieldname_str) == 0)
        {
            return elems_[i].str;
        }
    }
    return NULL;
}

/* get a specific scalari field */
int Params::getScalari(const char *fieldname)
{
    string fieldname_str(fieldname);
    for (int i = 0; i < elems_.size(); i++)
    {
        if (elems_[i].name.compare(fieldname_str) == 0)
        {
            return elems_[i].scalri;
        }
    }
    return 0;
}

/* get a specific scalarf field */
float Params::getScalarf(const char *fieldname)
{
    string fieldname_str(fieldname);
    for (int i = 0; i < elems_.size(); i++)
    {
        if (elems_[i].name.compare(fieldname_str) == 0)
        {
            return elems_[i].scalrf;
        }
    }
    return 0;
}

/* get a specific vectori field */
vector<int> Params::getVectori(const char *fieldname)
{
    string fieldname_str(fieldname);
    for (int i = 0; i < elems_.size(); i++)
    {
        if (elems_[i].name.compare(fieldname_str) == 0)
        {
            return elems_[i].vectri;
        }
    }
    return vector<int>();
}

/* get a specific vectorf field */
vector<float> Params::getVectorf(const char *fieldname)
{
    string fieldname_str(fieldname);
    for (int i = 0; i < elems_.size(); i++)
    {
        if (elems_[i].name.compare(fieldname_str) == 0)
        {
            return elems_[i].vectrf;
        }
    }
    return vector<float>();
}

/* get number of values in a specific field */
int Params::getNumel(const char *fieldname)
{
    string fieldname_str(fieldname);
    vector<element>::iterator it;
    for (int i = 0; i < elems_.size(); i++)
    {
        if (elems_[i].name.compare(fieldname_str) == 0)
        {
            if (elems_[i].type == STRING)
            {
                return 1;
            }
            else if (elems_[i].type == SCALARI || elems_[i].type == SCALARF)
            {
                return 1;
            }
            else if (elems_[i].type == VECTORI)
            {
                return elems_[i].vectri.size();
            }
            else if (elems_[i].type == VECTORF)
            {
                return elems_[i].vectrf.size();
            }
            return 0;
        }
    }
    return 0;
}
