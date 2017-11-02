/* ---------------------------------------------------------------------
 * Functions representing RHS, physical parameters, boundary conditions and
 * the true solution.
 * ---------------------------------------------------------------------
 *
 * Author: Eldar Khattatov, University of Pittsburgh, 2016 - 2017
 */

#ifndef ELASTICITY_MFEDD_DATA_H
#define ELASTICITY_MFEDD_DATA_H

#include <deal.II/base/function.h>

namespace dd_elasticity
{
    using namespace dealii;

    // Lame parameters (lambda and mu)
    template <int dim>
    class LameParameters : public Function<dim>
    {
    public:
        LameParameters ()  : Function<dim>() {}
        virtual void vector_value (const Point<dim> &p,
                                   Vector<double>   &values) const;

        virtual void vector_value_list (const std::vector<Point<dim> > &points,
        std::vector<Vector<double> > &value_list) const;
    };

    template <int dim>
    void LameParameters<dim>::vector_value (const Point<dim> &p,
                                            Vector<double>   &values) const
    {
        Assert(values.size() == 2,
               ExcDimensionMismatch(values.size(),2));
        Assert(dim != 1, ExcNotImplemented());

        double x,y,z;
        x = p[0];
        y = p[1];

        if (dim == 3)
            z = p[2];

        switch (dim)
        {
            case 2:
                if (p[0] < 0.5 && p[0] >= 0) {
                    values(0) = 1.0;
                    values(1) = 1.0;
                } else {
                    values(0) = 10.0;
                    values(1) = 10.0;
                }
                break;
            case 3:
                values(0) = 100.0;
                values(1) = 100.0;
                break;
            default:
                Assert(false, ExcNotImplemented());
        }
    }

    template <int dim>
    void LameParameters<dim>::vector_value_list(const std::vector<Point<dim> > &points,
                                                std::vector<Vector<double> > &value_list) const
    {
        Assert(value_list.size() == points.size(),
               ExcDimensionMismatch(value_list.size(), points.size()));

        const unsigned int n_points = points.size();

        for (unsigned int p=0; p<n_points; ++p)
            LameParameters<dim>::vector_value(points[p], value_list[p]);
    }

    // Right hand side values, boundary conditions and exact solution
    template <int dim>
    class RightHandSide : public Function<dim>
    {
        public:
            RightHandSide () : Function<dim>(dim) {}

            virtual void vector_value (const Point<dim> &p,
                                       Vector<double>   &values) const;

            virtual void vector_value_list (const std::vector<Point<dim> >   &points,
            std::vector<Vector<double> > &value_list) const;
    };

    template <int dim>
    inline
    void RightHandSide<dim>::vector_value(const Point<dim> &p,
                                          Vector<double>   &values) const
    {
        Assert(values.size() == dim,
               ExcDimensionMismatch(values.size(),dim));
        Assert(dim != 1, ExcNotImplemented());

        double x = p[0];
        double y = p[1];
        double z;

        if (dim == 3)
            z = p[2];

        const LameParameters<dim> lame_function;
        Vector<double> vec(2);
        lame_function.vector_value(p,vec);

        const double lmbda = vec[0];
        const double mu = vec[1];

        switch (dim)
        {
            case 2:
                if (p[0] < 0.5 && p[0] >= 0.0)
                {
                    values(0) = mu*(y*y*y)*(sin(M_PI*x)*2.0-(M_PI*M_PI)*(x*x)*sin(M_PI*x)+M_PI*x*cos(M_PI*x)*4.0-2.0)*2.0+lmbda*(y*y)*(x*-6.0-y*2.0+x*sin(M_PI*x)*6.0+y*sin(M_PI*x)*2.0+M_PI*(x*x)*cos(M_PI*x)*3.0+M_PI*x*y*cos(M_PI*x)*4.0-(M_PI*M_PI)*(x*x)*y*sin(M_PI*x))+mu*x*y*(x*-2.0-y*2.0+x*sin(M_PI*x)*2.0+y*sin(M_PI*x)*2.0+M_PI*x*y*cos(M_PI*x))*3.0;
                    values(1) = mu*(y*y)*(x*-6.0-y*2.0+x*sin(M_PI*x)*6.0+y*sin(M_PI*x)*2.0+M_PI*(x*x)*cos(M_PI*x)*3.0+M_PI*x*y*cos(M_PI*x)*4.0-(M_PI*M_PI)*(x*x)*y*sin(M_PI*x))+mu*(x*x)*y*(sin(M_PI*x)-1.0)*1.2E1+lmbda*x*y*(x*-2.0-y*2.0+x*sin(M_PI*x)*2.0+y*sin(M_PI*x)*2.0+M_PI*x*y*cos(M_PI*x))*3.0;
                }
                else
                {
                    values(0) = mu*(y*pow(x*2.0+9.0,2.0)*(-3.0/4.0E2)-(y*y)*(x*(1.0/5.0E1)+9.0/1.0E2)*(3.0/2.0)+y*sin(M_PI*(x*(1.0/1.0E1)+9.0/2.0E1))*pow(x*2.0+9.0,2.0)*(3.0/4.0E2)+(y*y)*sin(M_PI*(x*(1.0/1.0E1)+9.0/2.0E1))*(x*(1.0/5.0E1)+9.0/1.0E2)*(3.0/2.0)+M_PI*(y*y)*cos(M_PI*(x*(1.0/1.0E1)+9.0/2.0E1))*pow(x*2.0+9.0,2.0)*3.75E-4)*2.0+lmbda*((y*y)*(x*(1.0/5.0E1)+9.0/1.0E2)*-3.0+(y*y*y)*sin(M_PI*(x*(1.0/1.0E1)+9.0/2.0E1))*(1.0/5.0E1)-(y*y*y)*(1.0/5.0E1)+(y*y)*sin(M_PI*(x*(1.0/1.0E1)+9.0/2.0E1))*(x*(1.0/5.0E1)+9.0/1.0E2)*3.0+M_PI*(y*y*y)*cos(M_PI*(x*(1.0/1.0E1)+9.0/2.0E1))*(x*(1.0/5.0E1)+9.0/1.0E2)*(1.0/5.0)+M_PI*(y*y)*cos(M_PI*(x*(1.0/1.0E1)+9.0/2.0E1))*pow(x*2.0+9.0,2.0)*7.5E-4-(M_PI*M_PI)*(y*y*y)*sin(M_PI*(x*(1.0/1.0E1)+9.0/2.0E1))*pow(x*2.0+9.0,2.0)*2.5E-5)+mu*((y*y*y)*sin(M_PI*(x*(1.0/1.0E1)+9.0/2.0E1))*(1.0/5.0E1)-(y*y*y)*(1.0/5.0E1)+M_PI*(y*y*y)*cos(M_PI*(x*(1.0/1.0E1)+9.0/2.0E1))*(x*(1.0/5.0E1)+9.0/1.0E2)*(1.0/5.0)-(M_PI*M_PI)*(y*y*y)*sin(M_PI*(x*(1.0/1.0E1)+9.0/2.0E1))*pow(x*2.0+9.0,2.0)*2.5E-5)*2.0;
                    values(1) = lmbda*(y*pow(x*2.0+9.0,2.0)*(-3.0/2.0E2)-(y*y)*(x*(1.0/5.0E1)+9.0/1.0E2)*3.0+y*sin(M_PI*(x*(1.0/1.0E1)+9.0/2.0E1))*pow(x*2.0+9.0,2.0)*(3.0/2.0E2)+(y*y)*sin(M_PI*(x*(1.0/1.0E1)+9.0/2.0E1))*(x*(1.0/5.0E1)+9.0/1.0E2)*3.0+M_PI*(y*y)*cos(M_PI*(x*(1.0/1.0E1)+9.0/2.0E1))*pow(x*2.0+9.0,2.0)*7.5E-4)+mu*((y*y)*(x*(1.0/5.0E1)+9.0/1.0E2)*(-3.0/2.0)+(y*y*y)*sin(M_PI*(x*(1.0/1.0E1)+9.0/2.0E1))*(1.0/1.0E2)-(y*y*y)*(1.0/1.0E2)+(y*y)*sin(M_PI*(x*(1.0/1.0E1)+9.0/2.0E1))*(x*(1.0/5.0E1)+9.0/1.0E2)*(3.0/2.0)+M_PI*(y*y*y)*cos(M_PI*(x*(1.0/1.0E1)+9.0/2.0E1))*(x*(1.0/5.0E1)+9.0/1.0E2)*(1.0/1.0E1)+M_PI*(y*y)*cos(M_PI*(x*(1.0/1.0E1)+9.0/2.0E1))*pow(x*2.0+9.0,2.0)*3.75E-4-(M_PI*M_PI)*(y*y*y)*sin(M_PI*(x*(1.0/1.0E1)+9.0/2.0E1))*pow(x*2.0+9.0,2.0)*1.25E-5)*2.0+mu*y*pow(x*2.0+9.0,2.0)*(sin(M_PI*(x*(1.0/1.0E1)+9.0/2.0E1))-1.0)*(3.0/1.0E2);
                }
                break;
            case 3:
                values(0) = mu*(exp(x)*(cos(M_PI/12.0)-1.0)*(1.0/2.0)+(M_PI*M_PI)*sin(M_PI*y)*sin(M_PI*z)*(exp(x)*(1.0/1.0E1)-1.0/1.0E1)*(1.0/2.0))*-4.0+lmbda*exp(x)*(cos(M_PI/12.0)*-2.0E1+sin(M_PI*y)*sin(M_PI*z)+2.0E1)*(1.0/1.0E1)+mu*exp(x)*sin(M_PI*y)*sin(M_PI*z)*(1.0/5.0);
                values(1) = mu*(exp(x)*(y-cos(M_PI/12.0)*(y-1.0/2.0)+sin(M_PI/12.0)*(z-1.0/2.0)-1.0/2.0)*(1.0/2.0)+M_PI*exp(x)*cos(M_PI*y)*sin(M_PI*z)*(1.0/2.0E1))*2.0+M_PI*lmbda*exp(x)*cos(M_PI*y)*sin(M_PI*z)*(1.0/1.0E1);
                values(2) = mu*(exp(x)*(-z+cos(M_PI/12.0)*(z-1.0/2.0)+sin(M_PI/12.0)*(y-1.0/2.0)+1.0/2.0)*(1.0/2.0)-M_PI*exp(x)*cos(M_PI*z)*sin(M_PI*y)*(1.0/2.0E1))*-2.0+M_PI*lmbda*exp(x)*cos(M_PI*z)*sin(M_PI*y)*(1.0/1.0E1);
                break;
            default:
                Assert(false, ExcNotImplemented());
        }
    }

    template <int dim>
    void RightHandSide<dim>::vector_value_list(const std::vector<Point<dim> > &points,
                                               std::vector<Vector<double> >   &value_list) const
    {
        Assert(value_list.size() == points.size(),
               ExcDimensionMismatch(value_list.size(), points.size()));

        const unsigned int n_points = points.size();

        for (unsigned int p=0; p<n_points; ++p)
            RightHandSide<dim>::vector_value(points[p], value_list[p]);
    }

    // Boundary conditions (natural)
    template <int dim>
    class DisplacementBoundaryValues : public Function<dim>
    {
    public:
        DisplacementBoundaryValues() : Function<dim>(dim) {}

        virtual void vector_value (const Point<dim> &p,
                                   Vector<double>   &values) const;
        virtual void vector_value_list (const std::vector<Point<dim> >   &points,
        std::vector<Vector<double> > &value_list) const;
    };

    template <int dim>
    void DisplacementBoundaryValues<dim>::vector_value (const Point<dim> &p,
                                                        Vector<double>   &values) const
    {
        double x = p[0];
        double y = p[1];
        double z;

        if (dim == 3)
            z = p[2];

        const LameParameters<dim> lame_function;
        Vector<double> vec(2);
        lame_function.vector_value(p,vec);

        const double lmbda = vec[0];
        const double mu = vec[1];

        switch (dim)
        {
            case 2:
                if (p[0] < 0.5 && p[0] >= 0.0)
                {
                    values(0) = (x*x)*(y*y*y)-(x*x)*(y*y*y)*sin(M_PI*x);
                    values(1) = (x*x)*(y*y*y)-(x*x)*(y*y*y)*sin(M_PI*x);
                }
                else
                {
                    values(0) = (y*y*y)*pow(x*(1.0/1.0E1)+9.0/2.0E1,2.0)-(y*y*y)*sin(M_PI*(x*(1.0/1.0E1)+9.0/2.0E1))*pow(x*(1.0/1.0E1)+9.0/2.0E1,2.0);
                    values(1) = (y*y*y)*pow(x*(1.0/1.0E1)+9.0/2.0E1,2.0)-(y*y*y)*sin(M_PI*(x*(1.0/1.0E1)+9.0/2.0E1))*pow(x*(1.0/1.0E1)+9.0/2.0E1,2.0);
                }
                break;
            case 3:
                values(0) = -sin(M_PI*y)*sin(M_PI*z)*(exp(x)*(1.0/1.0E1)-1.0/1.0E1);
                values(1) = -(exp(x)-1.0)*(y-cos(M_PI/12.0)*(y-1.0/2.0)+sin(M_PI/12.0)*(z-1.0/2.0)-1.0/2.0);
                values(2) = (exp(x)-1.0)*(-z+cos(M_PI/12.0)*(z-1.0/2.0)+sin(M_PI/12.0)*(y-1.0/2.0)+1.0/2.0);
                break;
            default:
                Assert(false, ExcNotImplemented());
        }
    }

    template <int dim>
    void DisplacementBoundaryValues<dim>::vector_value_list(const std::vector<Point<dim> > &points,
    std::vector<Vector<double> >   &value_list) const
    {
        Assert(value_list.size() == points.size(),
               ExcDimensionMismatch(value_list.size(), points.size()));

        const unsigned int n_points = points.size();

        for (unsigned int p=0; p<n_points; ++p)
            DisplacementBoundaryValues<dim>::vector_value(points[p], value_list[p]);
    }

    // Exact solution
    template <int dim>
    class ExactSolution : public Function<dim>
    {
    public:
        ExactSolution() : Function<dim>(dim*dim + dim + 0.5*dim*(dim-1)) {}

        virtual void vector_value (const Point<dim> &p,
                                   Vector<double>   &value) const;
        virtual void vector_gradient (const Point<dim> &p,
                                      std::vector<Tensor<1,dim,double> > &grads) const;
    };

    template <int dim>
    void
    ExactSolution<dim>::vector_value (const Point<dim> &p,
                                      Vector<double>   &values) const
    {
        double x = p[0];
        double y = p[1];
        double z;

        if (dim == 3)
            z = p[2];

        const LameParameters<dim> lame_function;
        Vector<double> vec(2);
        lame_function.vector_value(p,vec);

        const double lmbda = vec[0];
        const double mu = vec[1];
        switch (dim)
        {
            case 2:
                if (p[0] < 0.5 && p[0] >= 0.0)
                {
                    values(0) = mu*x*(y*y*y)*(sin(M_PI*x)*2.0+M_PI*x*cos(M_PI*x)-2.0)*-2.0-lmbda*x*(y*y)*(x*-3.0-y*2.0+x*sin(M_PI*x)*3.0+y*sin(M_PI*x)*2.0+M_PI*x*y*cos(M_PI*x));
                    values(1) = -mu*x*(y*y)*(x*-3.0-y*2.0+x*sin(M_PI*x)*3.0+y*sin(M_PI*x)*2.0+M_PI*x*y*cos(M_PI*x));
                    values(2) = -mu*x*(y*y)*(x*-3.0-y*2.0+x*sin(M_PI*x)*3.0+y*sin(M_PI*x)*2.0+M_PI*x*y*cos(M_PI*x));
                    values(3) = -lmbda*((x*x)*(y*y)*-3.0-x*(y*y*y)*2.0+x*(y*y*y)*sin(M_PI*x)*2.0+(x*x)*(y*y)*sin(M_PI*x)*3.0+M_PI*(x*x)*(y*y*y)*cos(M_PI*x))+mu*((x*x)*(y*y)*3.0-(x*x)*(y*y)*sin(M_PI*x)*3.0)*2.0;

                    values(4) = (x*x)*(y*y*y)-(x*x)*(y*y*y)*sin(M_PI*x);
                    values(5) = (x*x)*(y*y*y)-(x*x)*(y*y*y)*sin(M_PI*x);

                    values(6) = x*(y*y)*(x*3.0-y*2.0-x*sin(M_PI*x)*3.0+y*sin(M_PI*x)*2.0+M_PI*x*y*cos(M_PI*x))*(1.0/2.0);
                }
                else
                {
                    values(0) = -lmbda*(-(y*y*y)*(x*(1.0/5.0E1)+9.0/1.0E2)-(y*y)*pow(x*(1.0/1.0E1)+9.0/2.0E1,2.0)*3.0+(y*y)*sin(M_PI*(x*(1.0/1.0E1)+9.0/2.0E1))*pow(x*(1.0/1.0E1)+9.0/2.0E1,2.0)*3.0+(y*y*y)*sin(M_PI*(x*(1.0/1.0E1)+9.0/2.0E1))*(x*(1.0/5.0E1)+9.0/1.0E2)+M_PI*(y*y*y)*cos(M_PI*(x*(1.0/1.0E1)+9.0/2.0E1))*pow(x*(1.0/1.0E1)+9.0/2.0E1,2.0)*(1.0/1.0E1))-mu*(-(y*y*y)*(x*(1.0/5.0E1)+9.0/1.0E2)+(y*y*y)*sin(M_PI*(x*(1.0/1.0E1)+9.0/2.0E1))*(x*(1.0/5.0E1)+9.0/1.0E2)+M_PI*(y*y*y)*cos(M_PI*(x*(1.0/1.0E1)+9.0/2.0E1))*pow(x*(1.0/1.0E1)+9.0/2.0E1,2.0)*(1.0/1.0E1))*2.0;
                    values(1) = mu*((y*y*y)*(x*(1.0/5.0E1)+9.0/1.0E2)*(-1.0/2.0)-(y*y)*pow(x*2.0+9.0,2.0)*(3.0/8.0E2)+(y*y)*sin(M_PI*(x*(1.0/1.0E1)+9.0/2.0E1))*pow(x*2.0+9.0,2.0)*(3.0/8.0E2)+(y*y*y)*sin(M_PI*(x*(1.0/1.0E1)+9.0/2.0E1))*(x*(1.0/5.0E1)+9.0/1.0E2)*(1.0/2.0)+M_PI*(y*y*y)*cos(M_PI*(x*(1.0/1.0E1)+9.0/2.0E1))*pow(x*2.0+9.0,2.0)*1.25E-4)*-2.0;
                    values(2) = mu*((y*y*y)*(x*(1.0/5.0E1)+9.0/1.0E2)*(-1.0/2.0)-(y*y)*pow(x*2.0+9.0,2.0)*(3.0/8.0E2)+(y*y)*sin(M_PI*(x*(1.0/1.0E1)+9.0/2.0E1))*pow(x*2.0+9.0,2.0)*(3.0/8.0E2)+(y*y*y)*sin(M_PI*(x*(1.0/1.0E1)+9.0/2.0E1))*(x*(1.0/5.0E1)+9.0/1.0E2)*(1.0/2.0)+M_PI*(y*y*y)*cos(M_PI*(x*(1.0/1.0E1)+9.0/2.0E1))*pow(x*2.0+9.0,2.0)*1.25E-4)*-2.0;
                    values(3) = -lmbda*(-(y*y*y)*(x*(1.0/5.0E1)+9.0/1.0E2)-(y*y)*pow(x*2.0+9.0,2.0)*(3.0/4.0E2)+(y*y)*sin(M_PI*(x*(1.0/1.0E1)+9.0/2.0E1))*pow(x*2.0+9.0,2.0)*(3.0/4.0E2)+(y*y*y)*sin(M_PI*(x*(1.0/1.0E1)+9.0/2.0E1))*(x*(1.0/5.0E1)+9.0/1.0E2)+M_PI*(y*y*y)*cos(M_PI*(x*(1.0/1.0E1)+9.0/2.0E1))*pow(x*2.0+9.0,2.0)*2.5E-4)-mu*(y*y)*pow(x*2.0+9.0,2.0)*(sin(M_PI*(x*(1.0/1.0E1)+9.0/2.0E1))-1.0)*(3.0/2.0E2);

                    values(4) = (y*y*y)*pow(x*(1.0/1.0E1)+9.0/2.0E1,2.0)-(y*y*y)*sin(M_PI*(x*(1.0/1.0E1)+9.0/2.0E1))*pow(x*(1.0/1.0E1)+9.0/2.0E1,2.0);
                    values(5) = (y*y*y)*pow(x*(1.0/1.0E1)+9.0/2.0E1,2.0)-(y*y*y)*sin(M_PI*(x*(1.0/1.0E1)+9.0/2.0E1))*pow(x*(1.0/1.0E1)+9.0/2.0E1,2.0);

                    values(6) = (y*y*y)*(x*(1.0/5.0E1)+9.0/1.0E2)*(-1.0/2.0)+(y*y)*pow(x*2.0+9.0,2.0)*(3.0/8.0E2)-(y*y)*sin(M_PI*(x*(1.0/1.0E1)+9.0/2.0E1))*pow(x*2.0+9.0,2.0)*(3.0/8.0E2)+(y*y*y)*sin(M_PI*(x*(1.0/1.0E1)+9.0/2.0E1))*(x*(1.0/5.0E1)+9.0/1.0E2)*(1.0/2.0)+M_PI*(y*y*y)*cos(M_PI*(x*(1.0/1.0E1)+9.0/2.0E1))*pow(x*2.0+9.0,2.0)*1.25E-4;
                }
                break;
            case 3:
                values(0) = lmbda*((exp(x)-1.0)*(cos(M_PI/12.0)-1.0)*2.0-exp(x)*sin(M_PI*y)*sin(M_PI*z)*(1.0/1.0E1))-mu*exp(x)*sin(M_PI*y)*sin(M_PI*z)*(1.0/5.0);
                values(1) = mu*(exp(x)*(y-cos(M_PI/12.0)*(y-1.0/2.0)+sin(M_PI/12.0)*(z-1.0/2.0)-1.0/2.0)*(1.0/2.0)+M_PI*cos(M_PI*y)*sin(M_PI*z)*(exp(x)*(1.0/1.0E1)-1.0/1.0E1)*(1.0/2.0))*-2.0;
                values(2) = mu*(exp(x)*(-z+cos(M_PI/12.0)*(z-1.0/2.0)+sin(M_PI/12.0)*(y-1.0/2.0)+1.0/2.0)*(1.0/2.0)-M_PI*cos(M_PI*z)*sin(M_PI*y)*(exp(x)*(1.0/1.0E1)-1.0/1.0E1)*(1.0/2.0))*2.0;
                values(3) = mu*(exp(x)*(y-cos(M_PI/12.0)*(y-1.0/2.0)+sin(M_PI/12.0)*(z-1.0/2.0)-1.0/2.0)*(1.0/2.0)+M_PI*cos(M_PI*y)*sin(M_PI*z)*(exp(x)*(1.0/1.0E1)-1.0/1.0E1)*(1.0/2.0))*-2.0;
                values(4) = lmbda*((exp(x)-1.0)*(cos(M_PI/12.0)-1.0)*2.0-exp(x)*sin(M_PI*y)*sin(M_PI*z)*(1.0/1.0E1))+mu*(exp(x)-1.0)*(cos(M_PI/12.0)-1.0)*2.0;
                values(5) = 0;
                values(6) = mu*(exp(x)*(-z+cos(M_PI/12.0)*(z-1.0/2.0)+sin(M_PI/12.0)*(y-1.0/2.0)+1.0/2.0)*(1.0/2.0)-M_PI*cos(M_PI*z)*sin(M_PI*y)*(exp(x)*(1.0/1.0E1)-1.0/1.0E1)*(1.0/2.0))*2.0;
                values(7) = 0;
                values(8) = lmbda*((exp(x)-1.0)*(cos(M_PI/12.0)-1.0)*2.0-exp(x)*sin(M_PI*y)*sin(M_PI*z)*(1.0/1.0E1))+mu*(exp(x)-1.0)*(cos(M_PI/12.0)-1.0)*2.0;

                values(9) = -sin(M_PI*y)*sin(M_PI*z)*(exp(x)*(1.0/1.0E1)-1.0/1.0E1);
                values(10) = -(exp(x)-1.0)*(y-cos(M_PI/12.0)*(y-1.0/2.0)+sin(M_PI/12.0)*(z-1.0/2.0)-1.0/2.0);
                values(11) = (exp(x)-1.0)*(-z+cos(M_PI/12.0)*(z-1.0/2.0)+sin(M_PI/12.0)*(y-1.0/2.0)+1.0/2.0);

                values(12) = sin(M_PI/12.0)*(exp(x)-1.0);
                values(13) = exp(x)*(-z+cos(M_PI/12.0)*(z-1.0/2.0)+sin(M_PI/12.0)*(y-1.0/2.0)+1.0/2.0)*(-1.0/2.0)-M_PI*cos(M_PI*z)*sin(M_PI*y)*(exp(x)*(1.0/1.0E1)-1.0/1.0E1)*(1.0/2.0);
                values(14) = exp(x)*(y-cos(M_PI/12.0)*(y-1.0/2.0)+sin(M_PI/12.0)*(z-1.0/2.0)-1.0/2.0)*(-1.0/2.0)+M_PI*cos(M_PI*y)*sin(M_PI*z)*(exp(x)*(1.0/1.0E1)-1.0/1.0E1)*(1.0/2.0);
                break;
            default:
                Assert(false, ExcNotImplemented());
        }
    }

    template <int dim>
    void
    ExactSolution<dim>::vector_gradient (const Point<dim> &p,
                                         std::vector<Tensor<1,dim,double> > &grads) const
    {
        double x = p[0];
        double y = p[1];
        double z;

        if (dim == 3)
        z = p[2];

        const LameParameters<dim> lame_function;
        Vector<double> vec(2);
        lame_function.vector_value(p,vec);

        const double lmbda = vec[0];
        const double mu = vec[1];

        int total_dim = dim*dim + dim + static_cast<int>(0.5*dim*(dim-1));
        Tensor<1,dim> tmp;
        switch (dim)
        {
        case 2:
            if (p[0] < 0.5 && p[0] >= 0.0)
            {
                tmp[0] = mu*(y*y*y)*(sin(M_PI*x)*2.0+M_PI*x*cos(M_PI*x)-2.0)*-2.0-lmbda*(y*y)*(x*-3.0-y*2.0+x*sin(M_PI*x)*3.0
                +y*sin(M_PI*x)*2.0+M_PI*x*y*cos(M_PI*x))-lmbda*x*(y*y)*(sin(M_PI*x)*3.0+M_PI*x*cos(M_PI*x)*3.0
                +M_PI*y*cos(M_PI*x)*3.0-(M_PI*M_PI)*x*y*sin(M_PI*x)-3.0)-mu*x*(y*y*y)*(M_PI*cos(M_PI*x)*3.0-(M_PI*M_PI)*x*sin(M_PI*x))*2.0;
                tmp[1] = -lmbda*x*(y*y)*(sin(M_PI*x)*2.0+M_PI*x*cos(M_PI*x)-2.0)-mu*x*(y*y)*(sin(M_PI*x)*2.0+M_PI*x*cos(M_PI*x)-2.0)*6.0
                -lmbda*x*y*(x*-3.0-y*2.0+x*sin(M_PI*x)*3.0+y*sin(M_PI*x)*2.0+M_PI*x*y*cos(M_PI*x))*2.0;
                grads[0] = tmp;

                tmp[0] = -mu*(y*y)*(x*-3.0-y*2.0+x*sin(M_PI*x)*3.0+y*sin(M_PI*x)*2.0+M_PI*x*y*cos(M_PI*x))-mu*x*(y*y)*(sin(M_PI*x)*3.0
                +M_PI*x*cos(M_PI*x)*3.0+M_PI*y*cos(M_PI*x)*3.0-(M_PI*M_PI)*x*y*sin(M_PI*x)-3.0);
                tmp[1] = -mu*x*(y*y)*(sin(M_PI*x)*2.0+M_PI*x*cos(M_PI*x)-2.0)-mu*x*y*(x*-3.0-y*2.0+x*sin(M_PI*x)*3.0+y*sin(M_PI*x)*2.0
                +M_PI*x*y*cos(M_PI*x))*2.0;
                grads[1] = tmp;
                tmp[0] = -mu*(y*y)*(x*-3.0-y*2.0+x*sin(M_PI*x)*3.0+y*sin(M_PI*x)*2.0+M_PI*x*y*cos(M_PI*x))-mu*x*(y*y)*(sin(M_PI*x)*3.0
                +M_PI*x*cos(M_PI*x)*3.0+M_PI*y*cos(M_PI*x)*3.0-(M_PI*M_PI)*x*y*sin(M_PI*x)-3.0);
                tmp[1] = -mu*x*(y*y)*(sin(M_PI*x)*2.0+M_PI*x*cos(M_PI*x)-2.0)-mu*x*y*(x*-3.0-y*2.0+x*sin(M_PI*x)*3.0+y*sin(M_PI*x)*2.0
                +M_PI*x*y*cos(M_PI*x))*2.0;
                grads[2] = tmp;

                tmp[0] = -lmbda*(x*(y*y)*-6.0+(y*y*y)*sin(M_PI*x)*2.0-(y*y*y)*2.0+x*(y*y)*sin(M_PI*x)*6.0-(M_PI*M_PI)*(x*x)*(y*y*y)*sin(M_PI*x)
                +M_PI*x*(y*y*y)*cos(M_PI*x)*4.0+M_PI*(x*x)*(y*y)*cos(M_PI*x)*3.0)-mu*(x*(y*y)*-6.0+x*(y*y)*sin(M_PI*x)*6.0+M_PI*(x*x)*(y*y)*cos(M_PI*x)*3.0)*2.0;
                tmp[1] = -lmbda*(x*(y*y)*-6.0-(x*x)*y*6.0+x*(y*y)*sin(M_PI*x)*6.0+(x*x)*y*sin(M_PI*x)*6.0+M_PI*(x*x)*(y*y)*cos(M_PI*x)*3.0)
                +mu*((x*x)*y*6.0-(x*x)*y*sin(M_PI*x)*6.0)*2.0;
                grads[3] = tmp;
            }
            else
            {
                tmp[0] = -lmbda*((y*y)*(x*(1.0/5.0E1)+9.0/1.0E2)*-3.0+(y*y*y)*sin(M_PI*(x*(1.0/1.0E1)+9.0/2.0E1))*(1.0/5.0E1)-(y*y*y)*(1.0/5.0E1)
                +(y*y)*sin(M_PI*(x*(1.0/1.0E1)+9.0/2.0E1))*(x*(1.0/5.0E1)+9.0/1.0E2)*3.0+M_PI*(y*y*y)*cos(M_PI*(x*(1.0/1.0E1)+9.0/2.0E1))
                *(x*(1.0/5.0E1)+9.0/1.0E2)*(1.0/5.0)+M_PI*(y*y)*cos(M_PI*(x*(1.0/1.0E1)+9.0/2.0E1))*pow(x*(1.0/1.0E1)+9.0/2.0E1,2.0)*(3.0/1.0E1)
                -(M_PI*M_PI)*(y*y*y)*sin(M_PI*(x*(1.0/1.0E1)+9.0/2.0E1))*pow(x*(1.0/1.0E1)+9.0/2.0E1,2.0)*(1.0/1.0E2))
                -mu*((y*y*y)*sin(M_PI*(x*(1.0/1.0E1)+9.0/2.0E1))*(1.0/5.0E1)-(y*y*y)*(1.0/5.0E1)+M_PI*(y*y*y)*cos(M_PI*(x*(1.0/1.0E1)+9.0/2.0E1))
                *(x*(1.0/5.0E1)+9.0/1.0E2)*(1.0/5.0)-(M_PI*M_PI)*(y*y*y)*sin(M_PI*(x*(1.0/1.0E1)+9.0/2.0E1))*pow(x*(1.0/1.0E1)+9.0/2.0E1,2.0)*(1.0/1.0E2))*2.0;
                tmp[1] = -lmbda*(y*pow(x*(1.0/1.0E1)+9.0/2.0E1,2.0)*-6.0-(y*y)*(x*(1.0/5.0E1)+9.0/1.0E2)*3.0+y*sin(M_PI*(x*(1.0/1.0E1)+9.0/2.0E1))
                *pow(x*(1.0/1.0E1)+9.0/2.0E1,2.0)*6.0+(y*y)*sin(M_PI*(x*(1.0/1.0E1)+9.0/2.0E1))*(x*(1.0/5.0E1)+9.0/1.0E2)*3.0+M_PI*(y*y)
                *cos(M_PI*(x*(1.0/1.0E1)+9.0/2.0E1))*pow(x*(1.0/1.0E1)+9.0/2.0E1,2.0)*(3.0/1.0E1))-mu*((y*y)*(x*(1.0/5.0E1)+9.0/1.0E2)*-3.0
                +(y*y)*sin(M_PI*(x*(1.0/1.0E1)+9.0/2.0E1))*(x*(1.0/5.0E1)+9.0/1.0E2)*3.0+M_PI*(y*y)*cos(M_PI*(x*(1.0/1.0E1)+9.0/2.0E1))
                *pow(x*(1.0/1.0E1)+9.0/2.0E1,2.0)*(3.0/1.0E1))*2.0;
                grads[0] = tmp;

                tmp[0] = mu*((y*y)*(x*8.0+3.6E1)*(-3.0/8.0E2)+(y*y*y)*sin(M_PI*(x*(1.0/1.0E1)+9.0/2.0E1))*(1.0/1.0E2)-(y*y*y)*(1.0/1.0E2)
                +(y*y)*sin(M_PI*(x*(1.0/1.0E1)+9.0/2.0E1))*(x*8.0+3.6E1)*(3.0/8.0E2)+M_PI*(y*y*y)*cos(M_PI*(x*(1.0/1.0E1)+9.0/2.0E1))
                *(x*8.0+3.6E1)*1.25E-4+M_PI*(y*y*y)*cos(M_PI*(x*(1.0/1.0E1)+9.0/2.0E1))*(x*(1.0/5.0E1)+9.0/1.0E2)*(1.0/2.0E1)+M_PI*(y*y)
                *cos(M_PI*(x*(1.0/1.0E1)+9.0/2.0E1))*pow(x*2.0+9.0,2.0)*3.75E-4-(M_PI*M_PI)*(y*y*y)*sin(M_PI*(x*(1.0/1.0E1)+9.0/2.0E1))
                *pow(x*2.0+9.0,2.0)*1.25E-5)*-2.0;
                tmp[1] = mu*(y*pow(x*2.0+9.0,2.0)*(-3.0/4.0E2)-(y*y)*(x*(1.0/5.0E1)+9.0/1.0E2)*(3.0/2.0)+y*sin(M_PI*(x*(1.0/1.0E1)+9.0/2.0E1))
                *pow(x*2.0+9.0,2.0)*(3.0/4.0E2)+(y*y)*sin(M_PI*(x*(1.0/1.0E1)+9.0/2.0E1))*(x*(1.0/5.0E1)+9.0/1.0E2)*(3.0/2.0)+M_PI*(y*y)
                *cos(M_PI*(x*(1.0/1.0E1)+9.0/2.0E1))*pow(x*2.0+9.0,2.0)*3.75E-4)*-2.0;
                grads[1] = tmp;

                tmp[0] = mu*((y*y)*(x*8.0+3.6E1)*(-3.0/8.0E2)+(y*y*y)*sin(M_PI*(x*(1.0/1.0E1)+9.0/2.0E1))*(1.0/1.0E2)-(y*y*y)*(1.0/1.0E2)
                +(y*y)*sin(M_PI*(x*(1.0/1.0E1)+9.0/2.0E1))*(x*8.0+3.6E1)*(3.0/8.0E2)+M_PI*(y*y*y)*cos(M_PI*(x*(1.0/1.0E1)+9.0/2.0E1))
                *(x*8.0+3.6E1)*1.25E-4+M_PI*(y*y*y)*cos(M_PI*(x*(1.0/1.0E1)+9.0/2.0E1))*(x*(1.0/5.0E1)+9.0/1.0E2)*(1.0/2.0E1)+M_PI*(y*y)
                *cos(M_PI*(x*(1.0/1.0E1)+9.0/2.0E1))*pow(x*2.0+9.0,2.0)*3.75E-4-(M_PI*M_PI)*(y*y*y)*sin(M_PI*(x*(1.0/1.0E1)+9.0/2.0E1))
                *pow(x*2.0+9.0,2.0)*1.25E-5)*-2.0;
                tmp[1] = mu*(y*pow(x*2.0+9.0,2.0)*(-3.0/4.0E2)-(y*y)*(x*(1.0/5.0E1)+9.0/1.0E2)*(3.0/2.0)+y*sin(M_PI*(x*(1.0/1.0E1)+9.0/2.0E1))
                *pow(x*2.0+9.0,2.0)*(3.0/4.0E2)+(y*y)*sin(M_PI*(x*(1.0/1.0E1)+9.0/2.0E1))*(x*(1.0/5.0E1)+9.0/1.0E2)*(3.0/2.0)+M_PI*(y*y)
                *cos(M_PI*(x*(1.0/1.0E1)+9.0/2.0E1))*pow(x*2.0+9.0,2.0)*3.75E-4)*-2.0;
                grads[2] = tmp;

                tmp[0] = -lmbda*((y*y)*(x*8.0+3.6E1)*(-3.0/4.0E2)+(y*y*y)*sin(M_PI*(x*(1.0/1.0E1)+9.0/2.0E1))*(1.0/5.0E1)-(y*y*y)*(1.0/5.0E1)
                +(y*y)*sin(M_PI*(x*(1.0/1.0E1)+9.0/2.0E1))*(x*8.0+3.6E1)*(3.0/4.0E2)+M_PI*(y*y*y)*cos(M_PI*(x*(1.0/1.0E1)+9.0/2.0E1))
                *(x*8.0+3.6E1)*2.5E-4+M_PI*(y*y*y)*cos(M_PI*(x*(1.0/1.0E1)+9.0/2.0E1))*(x*(1.0/5.0E1)+9.0/1.0E2)*(1.0/1.0E1)+M_PI*(y*y)
                *cos(M_PI*(x*(1.0/1.0E1)+9.0/2.0E1))*pow(x*2.0+9.0,2.0)*7.5E-4-(M_PI*M_PI)*(y*y*y)*sin(M_PI*(x*(1.0/1.0E1)+9.0/2.0E1))
                *pow(x*2.0+9.0,2.0)*2.5E-5)-mu*(y*y)*(x*8.0+3.6E1)*(sin(M_PI*(x*(1.0/1.0E1)+9.0/2.0E1))-1.0)*(3.0/2.0E2)-M_PI*mu*(y*y)
                *cos(M_PI*(x*(1.0/1.0E1)+9.0/2.0E1))*pow(x*2.0+9.0,2.0)*1.5E-3;
                tmp[1] = -lmbda*(y*pow(x*2.0+9.0,2.0)*(-3.0/2.0E2)-(y*y)*(x*(1.0/5.0E1)+9.0/1.0E2)*3.0+y*sin(M_PI*(x*(1.0/1.0E1)+9.0/2.0E1))
                *pow(x*2.0+9.0,2.0)*(3.0/2.0E2)+(y*y)*sin(M_PI*(x*(1.0/1.0E1)+9.0/2.0E1))*(x*(1.0/5.0E1)+9.0/1.0E2)*3.0+M_PI*(y*y)
                *cos(M_PI*(x*(1.0/1.0E1)+9.0/2.0E1))*pow(x*2.0+9.0,2.0)*7.5E-4)-mu*y*pow(x*2.0+9.0,2.0)*(sin(M_PI*(x*(1.0/1.0E1)+9.0/2.0E1))-1.0)*(3.0/1.0E2);
                grads[3] = tmp;
            }

            // The gradient for the rest is meaningless
            tmp[0] = 0.0;
            tmp[1] = 0.0;
            for (int k=dim*dim;k<total_dim;++k)
                grads[k] = tmp;

            break;
        case 3:
            tmp[0] = lmbda*(exp(x)*(cos(M_PI/12.0)-1.0)*2.0-exp(x)*sin(M_PI*y)*sin(M_PI*z)*(1.0/1.0E1))-mu*exp(x)*sin(M_PI*y)*sin(M_PI*z)*(1.0/5.0);
            tmp[1] = M_PI*lmbda*exp(x)*cos(M_PI*y)*sin(M_PI*z)*(-1.0/1.0E1)-M_PI*mu*exp(x)*cos(M_PI*y)*sin(M_PI*z)*(1.0/5.0);
            tmp[2] = M_PI*lmbda*exp(x)*cos(M_PI*z)*sin(M_PI*y)*(-1.0/1.0E1)-M_PI*mu*exp(x)*cos(M_PI*z)*sin(M_PI*y)*(1.0/5.0);
            grads[0] = tmp;

            tmp[0] = mu*(exp(x)*(y-cos(M_PI/12.0)*(y-1.0/2.0)+sin(M_PI/12.0)*(z-1.0/2.0)-1.0/2.0)*(1.0/2.0)+M_PI*exp(x)*cos(M_PI*y)*sin(M_PI*z)*(1.0/2.0E1))*-2.0;
            tmp[1] = mu*(exp(x)*(cos(M_PI/12.0)-1.0)*(1.0/2.0)+(M_PI*M_PI)*sin(M_PI*y)*sin(M_PI*z)*(exp(x)*(1.0/1.0E1)-1.0/1.0E1)*(1.0/2.0))*2.0;
            tmp[2] = mu*(sin(M_PI/12.0)*exp(x)*(1.0/2.0)+(M_PI*M_PI)*cos(M_PI*y)*cos(M_PI*z)*(exp(x)*(1.0/1.0E1)-1.0/1.0E1)*(1.0/2.0))*-2.0;
            grads[1] = tmp;

            tmp[0] = mu*(exp(x)*(-z+cos(M_PI/12.0)*(z-1.0/2.0)+sin(M_PI/12.0)*(y-1.0/2.0)+1.0/2.0)*(1.0/2.0)-M_PI*exp(x)*cos(M_PI*z)*sin(M_PI*y)*(1.0/2.0E1))*2.0;
            tmp[1] = mu*(sin(M_PI/12.0)*exp(x)*(1.0/2.0)-(M_PI*M_PI)*cos(M_PI*y)*cos(M_PI*z)*(exp(x)*(1.0/1.0E1)-1.0/1.0E1)*(1.0/2.0))*2.0;
            tmp[2] = mu*(exp(x)*(cos(M_PI/12.0)-1.0)*(1.0/2.0)+(M_PI*M_PI)*sin(M_PI*y)*sin(M_PI*z)*(exp(x)*(1.0/1.0E1)-1.0/1.0E1)*(1.0/2.0))*2.0;
            grads[2] = tmp;

            tmp[0] = mu*(exp(x)*(y-cos(M_PI/12.0)*(y-1.0/2.0)+sin(M_PI/12.0)*(z-1.0/2.0)-1.0/2.0)*(1.0/2.0)+M_PI*exp(x)*cos(M_PI*y)*sin(M_PI*z)*(1.0/2.0E1))*-2.0;
            tmp[1] = mu*(exp(x)*(cos(M_PI/12.0)-1.0)*(1.0/2.0)+(M_PI*M_PI)*sin(M_PI*y)*sin(M_PI*z)*(exp(x)*(1.0/1.0E1)-1.0/1.0E1)*(1.0/2.0))*2.0;
            tmp[2] = mu*(sin(M_PI/12.0)*exp(x)*(1.0/2.0)+(M_PI*M_PI)*cos(M_PI*y)*cos(M_PI*z)*(exp(x)*(1.0/1.0E1)-1.0/1.0E1)*(1.0/2.0))*-2.0;
            grads[3] = tmp;

            tmp[0] = lmbda*(exp(x)*(cos(M_PI/12.0)-1.0)*2.0-exp(x)*sin(M_PI*y)*sin(M_PI*z)*(1.0/1.0E1))+mu*exp(x)*(cos(M_PI/12.0)-1.0)*2.0;
            tmp[1] = M_PI*lmbda*exp(x)*cos(M_PI*y)*sin(M_PI*z)*(-1.0/1.0E1);
            tmp[2] = M_PI*lmbda*exp(x)*cos(M_PI*z)*sin(M_PI*y)*(-1.0/1.0E1);
            grads[4] = tmp;

            tmp[0] = 0;
            tmp[1] = 0;
            tmp[2] = 0;
            grads[5] = tmp;

            tmp[0] = mu*(exp(x)*(-z+cos(M_PI/12.0)*(z-1.0/2.0)+sin(M_PI/12.0)*(y-1.0/2.0)+1.0/2.0)*(1.0/2.0)-M_PI*exp(x)*cos(M_PI*z)*sin(M_PI*y)*(1.0/2.0E1))*2.0;
            tmp[1] = mu*(sin(M_PI/12.0)*exp(x)*(1.0/2.0)-(M_PI*M_PI)*cos(M_PI*y)*cos(M_PI*z)*(exp(x)*(1.0/1.0E1)-1.0/1.0E1)*(1.0/2.0))*2.0;
            tmp[2] = mu*(exp(x)*(cos(M_PI/12.0)-1.0)*(1.0/2.0)+(M_PI*M_PI)*sin(M_PI*y)*sin(M_PI*z)*(exp(x)*(1.0/1.0E1)-1.0/1.0E1)*(1.0/2.0))*2.0;
            grads[6] = tmp;

            tmp[0] = 0;
            tmp[1] = 0;
            tmp[2] = 0;
            grads[7] = tmp;

            tmp[0] = lmbda*(exp(x)*(cos(M_PI/12.0)-1.0)*2.0-exp(x)*sin(M_PI*y)*sin(M_PI*z)*(1.0/1.0E1))+mu*exp(x)*(cos(M_PI/12.0)-1.0)*2.0;
            tmp[1] = M_PI*lmbda*exp(x)*cos(M_PI*y)*sin(M_PI*z)*(-1.0/1.0E1);
            tmp[2] = M_PI*lmbda*exp(x)*cos(M_PI*z)*sin(M_PI*y)*(-1.0/1.0E1);
            grads[8] = tmp;

            // The gradient for the rest is meaningless
            tmp[0] = 0.0;
            tmp[1] = 0.0;
            tmp[2] = 0.0;
            for (int k=dim*dim;k<total_dim;++k)
                grads[k] = tmp;

            break;
        default:
            Assert(false, ExcNotImplemented());
        }
    }

//     // First and second Lame parameters
//     template <int dim>
//     class LameFirstParameter : public Function<dim>
//     {
//     public:
//         LameFirstParameter ()  : Function<dim>() {}
//         virtual double value (const Point<dim>   &p,
//                               const unsigned int  component = 0) const;
//         virtual void value_list (const std::vector<Point<dim> > &points,
//                                  std::vector<double>            &values,
//                                  const unsigned int              component = 0) const;
//     };

//     template <int dim>
//     double LameFirstParameter<dim>::value (const Point<dim> &p,
//                                            const unsigned int /* component */) const
//     {
//         const double nu = 0.2;
//         double x,y,z;
//         x = p[0];
//         y = p[1];

//         if (dim == 3)
//             z = p[2];

//         switch (dim)
//         {
//             case 2:
//                 return (sin(3.0*M_PI*x)*sin(3.0*M_PI*y)+5.0)*nu/((1.0-nu)*(1.0-2.0*nu));
//                 break;
//             case 3:
//                 //return exp(4*x)*nu/((1.0-nu)*(1.0-2.0*nu));
//                 return 100.0;
//                 break;
//             default:
//             Assert(false, ExcNotImplemented());
//         }

//     }

//     template <int dim>
//     void LameFirstParameter<dim>::value_list(const std::vector<Point<dim> > &points,
//                                              std::vector<double> &values,
//                                              const unsigned int component) const
//     {
//         Assert(values.size() == points.size(),
//                ExcDimensionMismatch(values.size(), points.size()));

//         const unsigned int n_points = points.size();

//         for (unsigned int p=0; p<n_points; ++p)
//             values[p] = LameFirstParameter<dim>::value(points[p]);
//     }

//     template <int dim>
//     class LameSecondParameter : public Function<dim>
//     {
//     public:
//         LameSecondParameter ()  : Function<dim>() {}
//         virtual double value (const Point<dim>   &p,
//                               const unsigned int  component = 0) const;
//         virtual void value_list (const std::vector<Point<dim> > &points,
//                                  std::vector<double>            &values,
//                                  const unsigned int              component = 0) const;
//     };

//     template <int dim>
//     double LameSecondParameter<dim>::value (const Point<dim> &p,
//                                             const unsigned int /* component */) const
//     {
//         const double nu = 0.2;
//         double x,y,z;
//         x = p[0];
//         y = p[1];

//         if (dim == 3)
//             z = p[2];

//         switch (dim)
//         {
//             case 2:
//                 return (sin(3.0*M_PI*x)*sin(3.0*M_PI*y)+5.0)/(2.0*(1.0+nu));
//                 break;
//             case 3:
//                 //return exp(4*x)/(2.0*(1.0+nu));
//                 return 100.0;
//                 break;
//             default:
//             Assert(false, ExcNotImplemented());
//         }
//     }

//     template <int dim>
//     void LameSecondParameter<dim>::value_list(const std::vector<Point<dim> > &points,
//                                               std::vector<double> &values,
//                                               const unsigned int component) const
//     {
//         Assert(values.size() == points.size(),
//                ExcDimensionMismatch(values.size(), points.size()));

//         const unsigned int n_points = points.size();

//         for (unsigned int p=0; p<n_points; ++p)
//             values[p] = LameSecondParameter<dim>::value(points[p]);
//     }

//     // Right hand side values, boundary conditions and exact solution
//     template <int dim>
//     class RightHandSide : public Function<dim>
//     {
//     public:
//         RightHandSide () : Function<dim>(dim) {}

//         virtual void vector_value (const Point<dim> &p,
//                                    Vector<double>   &values) const;
//         virtual void vector_value_list (const std::vector<Point<dim> >   &points,
//                                         std::vector<Vector<double> > &value_list) const;
//     };

//     template <int dim>
//     inline
//     void RightHandSide<dim>::vector_value(const Point<dim> &p,
//                                           Vector<double>   &values) const
//     {
//         Assert(values.size() == dim,
//                ExcDimensionMismatch(values.size(),dim));
//         Assert(dim != 1, ExcNotImplemented());

//         double x = p[0];
//         double y = p[1];
//         double z;

//         if (dim == 3)
//             z = p[2];

//         const LameFirstParameter<dim> lmbda_function;
//         const LameSecondParameter<dim> mu_function;

//         const double lmbda = lmbda_function.value(p);
//         const double mu = mu_function.value(p);

//         switch (dim)
//         {
//             case 2:
//                 values(0) = -(sin(M_PI*x*3.0)*sin(M_PI*y*3.0)*(5.0/6.0)+2.5E1/6.0)*(x*(y*y*y*y)*6.0-(y*y)*sin(x*y)*cos(y)+2.0)+(sin(M_PI*x*3.0)*sin(M_PI*y*3.0)*(5.0/6.0)+2.5E1/6.0)*(sin(x*y)*sin(x)*(1.0/2.0)-(x*x*x)*(y*y)*1.2E1+sin(x*y)*cos(y)*(1.0/2.0)+x*sin(x*y)*cos(x)*(1.0/2.0)+x*cos(x*y)*sin(y)+(x*x)*sin(x*y)*cos(y)*(1.0/2.0)+x*y*cos(x*y)*sin(x)*(1.0/2.0))+(sin(M_PI*x*3.0)*sin(M_PI*y*3.0)*(5.0/1.2E1)+2.5E1/1.2E1)*(sin(x*y)*sin(x)-(x*x*x)*(y*y)*1.2E1-x*(y*y*y*y)*6.0+x*sin(x*y)*cos(x)+(y*y)*sin(x*y)*cos(y)+x*y*cos(x*y)*sin(x)-2.0)-M_PI*cos(M_PI*y*3.0)*sin(M_PI*x*3.0)*((x*x*x)*(y*y*y)*4.0-sin(x*y)*sin(y)*(1.0/2.0)+cos(x*y)*cos(x)*(1.0/2.0)+x*cos(x*y)*cos(y)*(1.0/2.0)-y*sin(x*y)*sin(x)*(1.0/2.0))*(5.0/2.0)-M_PI*cos(M_PI*x*3.0)*sin(M_PI*y*3.0)*(x*2.0+(x*x)*(y*y*y*y)*3.0+y*cos(x*y)*cos(y))*(5.0/2.0)-M_PI*cos(M_PI*x*3.0)*sin(M_PI*y*3.0)*(x*2.0+y*2.0+(x*x)*(y*y*y*y)*3.0+(x*x*x*x)*(y*y)*3.0+y*cos(x*y)*cos(y)-x*sin(x*y)*sin(x))*(5.0/4.0);
//                 values(1) = -(sin(M_PI*x*3.0)*sin(M_PI*y*3.0)*(5.0/6.0)+2.5E1/6.0)*((x*x*x*x)*y*6.0-(x*x)*cos(x*y)*sin(x)+2.0)+(sin(M_PI*x*3.0)*sin(M_PI*y*3.0)*(5.0/6.0)+2.5E1/6.0)*((x*x)*(y*y*y)*-1.2E1-cos(x*y)*cos(y)*(1.0/2.0)+cos(x*y)*sin(x)*(1.0/2.0)+y*sin(x*y)*cos(x)+y*cos(x*y)*sin(y)*(1.0/2.0)+(y*y)*cos(x*y)*sin(x)*(1.0/2.0)+x*y*sin(x*y)*cos(y)*(1.0/2.0))-(sin(M_PI*x*3.0)*sin(M_PI*y*3.0)*(5.0/1.2E1)+2.5E1/1.2E1)*((x*x)*(y*y*y)*1.2E1+(x*x*x*x)*y*6.0+cos(x*y)*cos(y)-y*cos(x*y)*sin(y)-(x*x)*cos(x*y)*sin(x)-x*y*sin(x*y)*cos(y)+2.0)-M_PI*cos(M_PI*x*3.0)*sin(M_PI*y*3.0)*((x*x*x)*(y*y*y)*4.0-sin(x*y)*sin(y)*(1.0/2.0)+cos(x*y)*cos(x)*(1.0/2.0)+x*cos(x*y)*cos(y)*(1.0/2.0)-y*sin(x*y)*sin(x)*(1.0/2.0))*(5.0/2.0)-M_PI*cos(M_PI*y*3.0)*sin(M_PI*x*3.0)*(y*2.0+(x*x*x*x)*(y*y)*3.0-x*sin(x*y)*sin(x))*(5.0/2.0)-M_PI*cos(M_PI*y*3.0)*sin(M_PI*x*3.0)*(x*2.0+y*2.0+(x*x)*(y*y*y*y)*3.0+(x*x*x*x)*(y*y)*3.0+y*cos(x*y)*cos(y)-x*sin(x*y)*sin(x))*(5.0/4.0);
//                 break;
//             case 3:
// //        values(0) = exp(x*4.0)*(cos(M_PI/12.0)*8.0E1+exp(x)*1.3E2-cos(M_PI/12.0)*exp(x)*1.3E2+(M_PI*M_PI)*sin(M_PI*y)*sin(M_PI*z)*3.0+exp(x)*sin(M_PI*y)*sin(M_PI*z)*2.0E1-(M_PI*M_PI)*exp(x)*sin(M_PI*y)*sin(M_PI*z)*3.0-8.0E1)*(1.0/3.6E1);
// //        values(1) = exp(x*4.0)*(exp(x)*7.5E1-cos(M_PI/12.0)*exp(x)*7.5E1+sin(M_PI/12.0)*exp(x)*7.5E1-y*exp(x)*1.5E2+cos(M_PI/12.0)*y*exp(x)*1.5E2-sin(M_PI/12.0)*z*exp(x)*1.5E2+M_PI*cos(M_PI*y)*sin(M_PI*z)*1.2E1-M_PI*exp(x)*cos(M_PI*y)*sin(M_PI*z)*1.7E1)*(-1.0/7.2E1);
// //        values(2) = exp(x*4.0)*(exp(x)*7.5E1-cos(M_PI/12.0)*exp(x)*7.5E1-sin(M_PI/12.0)*exp(x)*7.5E1-z*exp(x)*1.5E2+cos(M_PI/12.0)*z*exp(x)*1.5E2+sin(M_PI/12.0)*y*exp(x)*1.5E2+M_PI*cos(M_PI*z)*sin(M_PI*y)*1.2E1-M_PI*exp(x)*cos(M_PI*z)*sin(M_PI*y)*1.7E1)*(-1.0/7.2E1);
//                 values(0) = mu*(exp(x)*(cos(M_PI/12.0)-1.0)*(1.0/2.0)+(M_PI*M_PI)*sin(M_PI*y)*sin(M_PI*z)*(exp(x)*(1.0/1.0E1)-1.0/1.0E1)*(1.0/2.0))*-4.0+lmbda*exp(x)*(cos(M_PI/12.0)*-2.0E1+sin(M_PI*y)*sin(M_PI*z)+2.0E1)*(1.0/1.0E1)+mu*exp(x)*sin(M_PI*y)*sin(M_PI*z)*(1.0/5.0);
//                 values(1) = mu*(exp(x)*(y-cos(M_PI/12.0)*(y-1.0/2.0)+sin(M_PI/12.0)*(z-1.0/2.0)-1.0/2.0)*(1.0/2.0)+M_PI*exp(x)*cos(M_PI*y)*sin(M_PI*z)*(1.0/2.0E1))*2.0+M_PI*lmbda*exp(x)*cos(M_PI*y)*sin(M_PI*z)*(1.0/1.0E1);
//                 values(2) = mu*(exp(x)*(-z+cos(M_PI/12.0)*(z-1.0/2.0)+sin(M_PI/12.0)*(y-1.0/2.0)+1.0/2.0)*(1.0/2.0)-M_PI*exp(x)*cos(M_PI*z)*sin(M_PI*y)*(1.0/2.0E1))*-2.0+M_PI*lmbda*exp(x)*cos(M_PI*z)*sin(M_PI*y)*(1.0/1.0E1);
//                 break;
//             default:
//             Assert(false, ExcNotImplemented());
//         }
//     }

//     template <int dim>
//     void RightHandSide<dim>::vector_value_list(const std::vector<Point<dim> > &points,
//                                                std::vector<Vector<double> >   &value_list) const
//     {
//         Assert(value_list.size() == points.size(),
//                ExcDimensionMismatch(value_list.size(), points.size()));

//         const unsigned int n_points = points.size();

//         for (unsigned int p=0; p<n_points; ++p)
//             RightHandSide<dim>::vector_value(points[p], value_list[p]);
//     }

//     // Boundary conditions (natural)
//     template <int dim>
//     class DisplacementBoundaryValues : public Function<dim>
//     {
//     public:
//         DisplacementBoundaryValues() : Function<dim>(dim) {}

//         virtual void vector_value (const Point<dim> &p,
//                                    Vector<double>   &values) const;
//         virtual void vector_value_list (const std::vector<Point<dim> >   &points,
//                                         std::vector<Vector<double> > &value_list) const;
//     };

//     template <int dim>
//     void DisplacementBoundaryValues<dim>::vector_value (const Point<dim> &p,
//                                                         Vector<double>   &values) const
//     {
//         double x = p[0];
//         double y = p[1];
//         double z;

//         if (dim == 3)
//             z = p[2];

//         const LameFirstParameter<dim> lmbda_function;
//         const LameSecondParameter<dim> mu_function;

//         const double lmbda = lmbda_function.value(p);
//         const double mu = mu_function.value(p);

//         switch (dim)
//         {
//             case 2:
//                 values(0) = (x*x*x)*(y*y*y*y)+x*x+sin(x*y)*cos(y);
//                 values(1) = (x*x*x*x)*(y*y*y)+y*y+cos(x*y)*sin(x);
//                 break;
//             case 3:
// //        values(0) = -sin(M_PI*y)*sin(M_PI*z)*(exp(x)*(1.0/1.0E1)-1.0/1.0E1);
// //        values(1) = -(exp(x)-1.0)*(y-cos(M_PI/12.0)*(y-1.0/2.0)+sin(M_PI/12.0)*(z-1.0/2.0)-1.0/2.0);
// //        values(2) = (exp(x)-1.0)*(-z+cos(M_PI/12.0)*(z-1.0/2.0)+sin(M_PI/12.0)*(y-1.0/2.0)+1.0/2.0);
//                 values(0) = -sin(M_PI*y)*sin(M_PI*z)*(exp(x)*(1.0/1.0E1)-1.0/1.0E1);
//                 values(1) = -(exp(x)-1.0)*(y-cos(M_PI/12.0)*(y-1.0/2.0)+sin(M_PI/12.0)*(z-1.0/2.0)-1.0/2.0);
//                 values(2) = (exp(x)-1.0)*(-z+cos(M_PI/12.0)*(z-1.0/2.0)+sin(M_PI/12.0)*(y-1.0/2.0)+1.0/2.0);
//                 break;
//             default:
//             Assert(false, ExcNotImplemented());
//         }
//     }

//     template <int dim>
//     void DisplacementBoundaryValues<dim>::vector_value_list(const std::vector<Point<dim> > &points,
//                                                             std::vector<Vector<double> >   &value_list) const
//     {
//         Assert(value_list.size() == points.size(),
//                ExcDimensionMismatch(value_list.size(), points.size()));

//         const unsigned int n_points = points.size();

//         for (unsigned int p=0; p<n_points; ++p)
//             DisplacementBoundaryValues<dim>::vector_value(points[p], value_list[p]);
//     }

//     // Exact solution
//     template <int dim>
//     class ExactSolution : public Function<dim>
//     {
//     public:
//         ExactSolution() : Function<dim>(dim*dim + dim + (3-dim)*(dim-1) + (dim-2)*dim) {}

//         virtual void vector_value (const Point<dim> &p,
//                                    Vector<double>   &value) const;
//         virtual void vector_gradient (const Point<dim> &p,
//                                       std::vector<Tensor<1,dim,double> > &grads) const;
//     };

//     template <int dim>
//     void
//     ExactSolution<dim>::vector_value (const Point<dim> &p,
//                                       Vector<double>   &values) const
//     {
//         double x = p[0];
//         double y = p[1];
//         double z;

//         if (dim == 3)
//             z = p[2];

//         const LameFirstParameter<dim> lmbda_function;
//         const LameSecondParameter<dim> mu_function;

//         const double lmbda = lmbda_function.value(p);
//         const double mu = mu_function.value(p);

//         switch (dim)
//         {
//             case 2:
//                 // Stress Test 1
//                 values(0) = (sin(M_PI*x*3.0)*sin(M_PI*y*3.0)+5.0)*(x*6.0+y*2.0+(x*x)*(y*y*y*y)*9.0+(x*x*x*x)*(y*y)*3.0+y*cos(x*y)*cos(y)*3.0-x*sin(x*y)*sin(x))*(5.0/1.2E1);
//                 values(1) = (sin(M_PI*x*3.0)*sin(M_PI*y*3.0)*(5.0/6.0)+2.5E1/6.0)*((x*x*x)*(y*y*y)*4.0-sin(x*y)*sin(y)*(1.0/2.0)+cos(x*y)*cos(x)*(1.0/2.0)+x*cos(x*y)*cos(y)*(1.0/2.0)-y*sin(x*y)*sin(x)*(1.0/2.0));
//                 values(2) = (sin(M_PI*x*3.0)*sin(M_PI*y*3.0)*(5.0/6.0)+2.5E1/6.0)*((x*x*x)*(y*y*y)*4.0-sin(x*y)*sin(y)*(1.0/2.0)+cos(x*y)*cos(x)*(1.0/2.0)+x*cos(x*y)*cos(y)*(1.0/2.0)-y*sin(x*y)*sin(x)*(1.0/2.0));
//                 values(3) = (sin(M_PI*x*3.0)*sin(M_PI*y*3.0)+5.0)*(x*2.0+y*6.0+(x*x)*(y*y*y*y)*3.0+(x*x*x*x)*(y*y)*9.0+y*cos(x*y)*cos(y)-x*sin(x*y)*sin(x)*3.0)*(5.0/1.2E1);
//                 // Displacement
//                 values(4) = (x*x*x)*(y*y*y*y)+x*x+sin(x*y)*cos(y);
//                 values(5) = (x*x*x*x)*(y*y*y)+y*y+cos(x*y)*sin(x);
//                 // Rotation
//                 values(6) = sin(x*y)*sin(y)*(-1.0/2.0)-cos(x*y)*cos(x)*(1.0/2.0)+x*cos(x*y)*cos(y)*(1.0/2.0)+y*sin(x*y)*sin(x)*(1.0/2.0);
//                 break;
//             case 3:
// //        // Stress
// //        values(0) = exp(x*4.0)*(cos(M_PI/12.0)*5.0+exp(x)*5.0-cos(M_PI/12.0)*exp(x)*5.0+exp(x)*sin(M_PI*y)*sin(M_PI*z)-5.0)*(-1.0/9.0);
// //        values(1) = exp(x*4.0)*(exp(x)*(y-cos(M_PI/12.0)*(y-1.0/2.0)+sin(M_PI/12.0)*(z-1.0/2.0)-1.0/2.0)*(1.0/2.0)+M_PI*cos(M_PI*y)*sin(M_PI*z)*(exp(x)*(1.0/1.0E1)-1.0/1.0E1)*(1.0/2.0))*(-5.0/6.0);
// //        values(2) = exp(x*4.0)*(exp(x)*(-z+cos(M_PI/12.0)*(z-1.0/2.0)+sin(M_PI/12.0)*(y-1.0/2.0)+1.0/2.0)*(1.0/2.0)-M_PI*cos(M_PI*z)*sin(M_PI*y)*(exp(x)*(1.0/1.0E1)-1.0/1.0E1)*(1.0/2.0))*(5.0/6.0);
// //        values(3) = exp(x*4.0)*(exp(x)*(y-cos(M_PI/12.0)*(y-1.0/2.0)+sin(M_PI/12.0)*(z-1.0/2.0)-1.0/2.0)*(1.0/2.0)+M_PI*cos(M_PI*y)*sin(M_PI*z)*(exp(x)*(1.0/1.0E1)-1.0/1.0E1)*(1.0/2.0))*(-5.0/6.0);
// //        values(4) = exp(x*4.0)*((exp(x)-1.0)*(cos(M_PI/12.0)-1.0)*2.0-exp(x)*sin(M_PI*y)*sin(M_PI*z)*(1.0/1.0E1))*(5.0/1.8E1)+exp(x*4.0)*(exp(x)-1.0)*(cos(M_PI/12.0)-1.0)*(5.0/6.0);
// //        values(5) = 0;
// //        values(6) = exp(x*4.0)*(exp(x)*(-z+cos(M_PI/12.0)*(z-1.0/2.0)+sin(M_PI/12.0)*(y-1.0/2.0)+1.0/2.0)*(1.0/2.0)-M_PI*cos(M_PI*z)*sin(M_PI*y)*(exp(x)*(1.0/1.0E1)-1.0/1.0E1)*(1.0/2.0))*(5.0/6.0);
// //        values(7) = 0;
// //        values(8) = exp(x*4.0)*((exp(x)-1.0)*(cos(M_PI/12.0)-1.0)*2.0-exp(x)*sin(M_PI*y)*sin(M_PI*z)*(1.0/1.0E1))*(5.0/1.8E1)+exp(x*4.0)*(exp(x)-1.0)*(cos(M_PI/12.0)-1.0)*(5.0/6.0);
// //        // Displacement
// //        values(9) = -sin(M_PI*y)*sin(M_PI*z)*(exp(x)*(1.0/1.0E1)-1.0/1.0E1);
// //        values(10) = -(exp(x)-1.0)*(y-cos(M_PI/12.0)*(y-1.0/2.0)+sin(M_PI/12.0)*(z-1.0/2.0)-1.0/2.0);
// //        values(11) = (exp(x)-1.0)*(-z+cos(M_PI/12.0)*(z-1.0/2.0)+sin(M_PI/12.0)*(y-1.0/2.0)+1.0/2.0);
// //        // Rotation
// //        values(12) = sin(M_PI/12.0)*(exp(x)-1.0);
// //        values(13) = exp(x)*(-z+cos(M_PI/12.0)*(z-1.0/2.0)+sin(M_PI/12.0)*(y-1.0/2.0)+1.0/2.0)*(-1.0/2.0)-M_PI*cos(M_PI*z)*sin(M_PI*y)*(exp(x)*(1.0/1.0E1)-1.0/1.0E1)*(1.0/2.0);
// //        values(14) = exp(x)*(y-cos(M_PI/12.0)*(y-1.0/2.0)+sin(M_PI/12.0)*(z-1.0/2.0)-1.0/2.0)*(-1.0/2.0)+M_PI*cos(M_PI*y)*sin(M_PI*z)*(exp(x)*(1.0/1.0E1)-1.0/1.0E1)*(1.0/2.0);
//                 values(0) = lmbda*((exp(x)-1.0)*(cos(M_PI/12.0)-1.0)*2.0-exp(x)*sin(M_PI*y)*sin(M_PI*z)*(1.0/1.0E1))-mu*exp(x)*sin(M_PI*y)*sin(M_PI*z)*(1.0/5.0);
//                 values(1) = mu*(exp(x)*(y-cos(M_PI/12.0)*(y-1.0/2.0)+sin(M_PI/12.0)*(z-1.0/2.0)-1.0/2.0)*(1.0/2.0)+M_PI*cos(M_PI*y)*sin(M_PI*z)*(exp(x)*(1.0/1.0E1)-1.0/1.0E1)*(1.0/2.0))*-2.0;
//                 values(2) = mu*(exp(x)*(-z+cos(M_PI/12.0)*(z-1.0/2.0)+sin(M_PI/12.0)*(y-1.0/2.0)+1.0/2.0)*(1.0/2.0)-M_PI*cos(M_PI*z)*sin(M_PI*y)*(exp(x)*(1.0/1.0E1)-1.0/1.0E1)*(1.0/2.0))*2.0;
//                 values(3) = mu*(exp(x)*(y-cos(M_PI/12.0)*(y-1.0/2.0)+sin(M_PI/12.0)*(z-1.0/2.0)-1.0/2.0)*(1.0/2.0)+M_PI*cos(M_PI*y)*sin(M_PI*z)*(exp(x)*(1.0/1.0E1)-1.0/1.0E1)*(1.0/2.0))*-2.0;
//                 values(4) = lmbda*((exp(x)-1.0)*(cos(M_PI/12.0)-1.0)*2.0-exp(x)*sin(M_PI*y)*sin(M_PI*z)*(1.0/1.0E1))+mu*(exp(x)-1.0)*(cos(M_PI/12.0)-1.0)*2.0;
//                 values(5) = 0;
//                 values(6) = mu*(exp(x)*(-z+cos(M_PI/12.0)*(z-1.0/2.0)+sin(M_PI/12.0)*(y-1.0/2.0)+1.0/2.0)*(1.0/2.0)-M_PI*cos(M_PI*z)*sin(M_PI*y)*(exp(x)*(1.0/1.0E1)-1.0/1.0E1)*(1.0/2.0))*2.0;
//                 values(7) = 0;
//                 values(8) = lmbda*((exp(x)-1.0)*(cos(M_PI/12.0)-1.0)*2.0-exp(x)*sin(M_PI*y)*sin(M_PI*z)*(1.0/1.0E1))+mu*(exp(x)-1.0)*(cos(M_PI/12.0)-1.0)*2.0;

//                 values(9) = -sin(M_PI*y)*sin(M_PI*z)*(exp(x)*(1.0/1.0E1)-1.0/1.0E1);
//                 values(10) = -(exp(x)-1.0)*(y-cos(M_PI/12.0)*(y-1.0/2.0)+sin(M_PI/12.0)*(z-1.0/2.0)-1.0/2.0);
//                 values(11) = (exp(x)-1.0)*(-z+cos(M_PI/12.0)*(z-1.0/2.0)+sin(M_PI/12.0)*(y-1.0/2.0)+1.0/2.0);

//                 values(12) = sin(M_PI/12.0)*(exp(x)-1.0);
//                 values(13) = exp(x)*(-z+cos(M_PI/12.0)*(z-1.0/2.0)+sin(M_PI/12.0)*(y-1.0/2.0)+1.0/2.0)*(-1.0/2.0)-M_PI*cos(M_PI*z)*sin(M_PI*y)*(exp(x)*(1.0/1.0E1)-1.0/1.0E1)*(1.0/2.0);
//                 values(14) = exp(x)*(y-cos(M_PI/12.0)*(y-1.0/2.0)+sin(M_PI/12.0)*(z-1.0/2.0)-1.0/2.0)*(-1.0/2.0)+M_PI*cos(M_PI*y)*sin(M_PI*z)*(exp(x)*(1.0/1.0E1)-1.0/1.0E1)*(1.0/2.0);
//                 break;
//             default:
//             Assert(false, ExcNotImplemented());
//         }
//     }

//     template <int dim>
//     void
//     ExactSolution<dim>::vector_gradient (const Point<dim> &p,
//                                          std::vector<Tensor<1,dim,double> > &grads) const
//     {
//         double x = p[0];
//         double y = p[1];
//         double z;

//         if (dim == 3)
//             z = p[2];

//         const LameFirstParameter<dim> lmbda_function;
//         const LameSecondParameter<dim> mu_function;

//         const double lmbda = lmbda_function.value(p);
//         const double mu = mu_function.value(p);

//         int total_dim = dim*dim + dim + static_cast<int>(0.5*dim*(dim-1));
//         Tensor<1,dim> tmp;
//         switch (dim)
//         {
//             case 2:
//                 // sigma_11 Test1
//                 tmp[0] = (sin(M_PI*x*3.0)*sin(M_PI*y*3.0)+5.0)*(sin(x*y)*sin(x)-(x*x*x)*(y*y)*1.2E1-x*(y*y*y*y)*1.8E1+x*sin(x*y)*cos(x)+(y*y)*sin(x*y)*cos(y)*3.0+x*y*cos(x*y)*sin(x)-6.0)*(-5.0/1.2E1)+M_PI*cos(M_PI*x*3.0)*sin(M_PI*y*3.0)*(x*6.0+y*2.0+(x*x)*(y*y*y*y)*9.0+(x*x*x*x)*(y*y)*3.0+y*cos(x*y)*cos(y)*3.0-x*sin(x*y)*sin(x))*(5.0/4.0);
//                 tmp[1] = (sin(M_PI*x*3.0)*sin(M_PI*y*3.0)+5.0)*((x*x)*(y*y*y)*3.6E1+(x*x*x*x)*y*6.0+cos(x*y)*cos(y)*3.0-y*cos(x*y)*sin(y)*3.0-(x*x)*cos(x*y)*sin(x)-x*y*sin(x*y)*cos(y)*3.0+2.0)*(5.0/1.2E1)+M_PI*cos(M_PI*y*3.0)*sin(M_PI*x*3.0)*(x*6.0+y*2.0+(x*x)*(y*y*y*y)*9.0+(x*x*x*x)*(y*y)*3.0+y*cos(x*y)*cos(y)*3.0-x*sin(x*y)*sin(x))*(5.0/4.0);
//                 grads[0] = tmp;
//                 // sigma_12
//                 tmp[0] = -(sin(M_PI*x*3.0)*sin(M_PI*y*3.0)*(5.0/6.0)+2.5E1/6.0)*((x*x)*(y*y*y)*-1.2E1-cos(x*y)*cos(y)*(1.0/2.0)+cos(x*y)*sin(x)*(1.0/2.0)+y*sin(x*y)*cos(x)+y*cos(x*y)*sin(y)*(1.0/2.0)+(y*y)*cos(x*y)*sin(x)*(1.0/2.0)+x*y*sin(x*y)*cos(y)*(1.0/2.0))+M_PI*cos(M_PI*x*3.0)*sin(M_PI*y*3.0)*((x*x*x)*(y*y*y)*4.0-sin(x*y)*sin(y)*(1.0/2.0)+cos(x*y)*cos(x)*(1.0/2.0)+x*cos(x*y)*cos(y)*(1.0/2.0)-y*sin(x*y)*sin(x)*(1.0/2.0))*(5.0/2.0);
//                 tmp[1] = -(sin(M_PI*x*3.0)*sin(M_PI*y*3.0)*(5.0/6.0)+2.5E1/6.0)*(sin(x*y)*sin(x)*(1.0/2.0)-(x*x*x)*(y*y)*1.2E1+sin(x*y)*cos(y)*(1.0/2.0)+x*sin(x*y)*cos(x)*(1.0/2.0)+x*cos(x*y)*sin(y)+(x*x)*sin(x*y)*cos(y)*(1.0/2.0)+x*y*cos(x*y)*sin(x)*(1.0/2.0))+M_PI*cos(M_PI*y*3.0)*sin(M_PI*x*3.0)*((x*x*x)*(y*y*y)*4.0-sin(x*y)*sin(y)*(1.0/2.0)+cos(x*y)*cos(x)*(1.0/2.0)+x*cos(x*y)*cos(y)*(1.0/2.0)-y*sin(x*y)*sin(x)*(1.0/2.0))*(5.0/2.0);
//                 grads[1] = tmp;
//                 // sigma_21
//                 tmp[0] = -(sin(M_PI*x*3.0)*sin(M_PI*y*3.0)*(5.0/6.0)+2.5E1/6.0)*((x*x)*(y*y*y)*-1.2E1-cos(x*y)*cos(y)*(1.0/2.0)+cos(x*y)*sin(x)*(1.0/2.0)+y*sin(x*y)*cos(x)+y*cos(x*y)*sin(y)*(1.0/2.0)+(y*y)*cos(x*y)*sin(x)*(1.0/2.0)+x*y*sin(x*y)*cos(y)*(1.0/2.0))+M_PI*cos(M_PI*x*3.0)*sin(M_PI*y*3.0)*((x*x*x)*(y*y*y)*4.0-sin(x*y)*sin(y)*(1.0/2.0)+cos(x*y)*cos(x)*(1.0/2.0)+x*cos(x*y)*cos(y)*(1.0/2.0)-y*sin(x*y)*sin(x)*(1.0/2.0))*(5.0/2.0);
//                 tmp[1] = -(sin(M_PI*x*3.0)*sin(M_PI*y*3.0)*(5.0/6.0)+2.5E1/6.0)*(sin(x*y)*sin(x)*(1.0/2.0)-(x*x*x)*(y*y)*1.2E1+sin(x*y)*cos(y)*(1.0/2.0)+x*sin(x*y)*cos(x)*(1.0/2.0)+x*cos(x*y)*sin(y)+(x*x)*sin(x*y)*cos(y)*(1.0/2.0)+x*y*cos(x*y)*sin(x)*(1.0/2.0))+M_PI*cos(M_PI*y*3.0)*sin(M_PI*x*3.0)*((x*x*x)*(y*y*y)*4.0-sin(x*y)*sin(y)*(1.0/2.0)+cos(x*y)*cos(x)*(1.0/2.0)+x*cos(x*y)*cos(y)*(1.0/2.0)-y*sin(x*y)*sin(x)*(1.0/2.0))*(5.0/2.0);
//                 grads[2] = tmp;
//                 // sigma_22
//                 tmp[0] = (sin(M_PI*x*3.0)*sin(M_PI*y*3.0)+5.0)*(sin(x*y)*sin(x)*3.0-(x*x*x)*(y*y)*3.6E1-x*(y*y*y*y)*6.0+x*sin(x*y)*cos(x)*3.0+(y*y)*sin(x*y)*cos(y)+x*y*cos(x*y)*sin(x)*3.0-2.0)*(-5.0/1.2E1)+M_PI*cos(M_PI*x*3.0)*sin(M_PI*y*3.0)*(x*2.0+y*6.0+(x*x)*(y*y*y*y)*3.0+(x*x*x*x)*(y*y)*9.0+y*cos(x*y)*cos(y)-x*sin(x*y)*sin(x)*3.0)*(5.0/4.0);
//                 tmp[1] = (sin(M_PI*x*3.0)*sin(M_PI*y*3.0)+5.0)*((x*x)*(y*y*y)*1.2E1+(x*x*x*x)*y*1.8E1+cos(x*y)*cos(y)-y*cos(x*y)*sin(y)-(x*x)*cos(x*y)*sin(x)*3.0-x*y*sin(x*y)*cos(y)+6.0)*(5.0/1.2E1)+M_PI*cos(M_PI*y*3.0)*sin(M_PI*x*3.0)*(x*2.0+y*6.0+(x*x)*(y*y*y*y)*3.0+(x*x*x*x)*(y*y)*9.0+y*cos(x*y)*cos(y)-x*sin(x*y)*sin(x)*3.0)*(5.0/4.0);
//                 grads[3] = tmp;
//                 // The gradient for the rest is meaningless
//                 tmp[0] = 0.0;
//                 tmp[1] = 0.0;
//                 for (int k=dim*dim;k<total_dim;++k)
//                     grads[k] = tmp;

//                 break;
//             case 3:
//                 tmp[0] = lmbda*(exp(x)*(cos(M_PI/12.0)-1.0)*2.0-exp(x)*sin(M_PI*y)*sin(M_PI*z)*(1.0/1.0E1))-mu*exp(x)*sin(M_PI*y)*sin(M_PI*z)*(1.0/5.0);
//                 tmp[1] = M_PI*lmbda*exp(x)*cos(M_PI*y)*sin(M_PI*z)*(-1.0/1.0E1)-M_PI*mu*exp(x)*cos(M_PI*y)*sin(M_PI*z)*(1.0/5.0);
//                 tmp[2] = M_PI*lmbda*exp(x)*cos(M_PI*z)*sin(M_PI*y)*(-1.0/1.0E1)-M_PI*mu*exp(x)*cos(M_PI*z)*sin(M_PI*y)*(1.0/5.0);
//                 grads[0] = tmp;

//                 tmp[0] = mu*(exp(x)*(y-cos(M_PI/12.0)*(y-1.0/2.0)+sin(M_PI/12.0)*(z-1.0/2.0)-1.0/2.0)*(1.0/2.0)+M_PI*exp(x)*cos(M_PI*y)*sin(M_PI*z)*(1.0/2.0E1))*-2.0;
//                 tmp[1] = mu*(exp(x)*(cos(M_PI/12.0)-1.0)*(1.0/2.0)+(M_PI*M_PI)*sin(M_PI*y)*sin(M_PI*z)*(exp(x)*(1.0/1.0E1)-1.0/1.0E1)*(1.0/2.0))*2.0;
//                 tmp[2] = mu*(sin(M_PI/12.0)*exp(x)*(1.0/2.0)+(M_PI*M_PI)*cos(M_PI*y)*cos(M_PI*z)*(exp(x)*(1.0/1.0E1)-1.0/1.0E1)*(1.0/2.0))*-2.0;
//                 grads[1] = tmp;

//                 tmp[0] = mu*(exp(x)*(-z+cos(M_PI/12.0)*(z-1.0/2.0)+sin(M_PI/12.0)*(y-1.0/2.0)+1.0/2.0)*(1.0/2.0)-M_PI*exp(x)*cos(M_PI*z)*sin(M_PI*y)*(1.0/2.0E1))*2.0;
//                 tmp[1] = mu*(sin(M_PI/12.0)*exp(x)*(1.0/2.0)-(M_PI*M_PI)*cos(M_PI*y)*cos(M_PI*z)*(exp(x)*(1.0/1.0E1)-1.0/1.0E1)*(1.0/2.0))*2.0;
//                 tmp[2] = mu*(exp(x)*(cos(M_PI/12.0)-1.0)*(1.0/2.0)+(M_PI*M_PI)*sin(M_PI*y)*sin(M_PI*z)*(exp(x)*(1.0/1.0E1)-1.0/1.0E1)*(1.0/2.0))*2.0;
//                 grads[2] = tmp;

//                 tmp[0] = mu*(exp(x)*(y-cos(M_PI/12.0)*(y-1.0/2.0)+sin(M_PI/12.0)*(z-1.0/2.0)-1.0/2.0)*(1.0/2.0)+M_PI*exp(x)*cos(M_PI*y)*sin(M_PI*z)*(1.0/2.0E1))*-2.0;
//                 tmp[1] = mu*(exp(x)*(cos(M_PI/12.0)-1.0)*(1.0/2.0)+(M_PI*M_PI)*sin(M_PI*y)*sin(M_PI*z)*(exp(x)*(1.0/1.0E1)-1.0/1.0E1)*(1.0/2.0))*2.0;
//                 tmp[2] = mu*(sin(M_PI/12.0)*exp(x)*(1.0/2.0)+(M_PI*M_PI)*cos(M_PI*y)*cos(M_PI*z)*(exp(x)*(1.0/1.0E1)-1.0/1.0E1)*(1.0/2.0))*-2.0;
//                 grads[3] = tmp;

//                 tmp[0] = lmbda*(exp(x)*(cos(M_PI/12.0)-1.0)*2.0-exp(x)*sin(M_PI*y)*sin(M_PI*z)*(1.0/1.0E1))+mu*exp(x)*(cos(M_PI/12.0)-1.0)*2.0;
//                 tmp[1] = M_PI*lmbda*exp(x)*cos(M_PI*y)*sin(M_PI*z)*(-1.0/1.0E1);
//                 tmp[2] = M_PI*lmbda*exp(x)*cos(M_PI*z)*sin(M_PI*y)*(-1.0/1.0E1);
//                 grads[4] = tmp;

//                 tmp[0] = 0;
//                 tmp[1] = 0;
//                 tmp[2] = 0;
//                 grads[5] = tmp;

//                 tmp[0] = mu*(exp(x)*(-z+cos(M_PI/12.0)*(z-1.0/2.0)+sin(M_PI/12.0)*(y-1.0/2.0)+1.0/2.0)*(1.0/2.0)-M_PI*exp(x)*cos(M_PI*z)*sin(M_PI*y)*(1.0/2.0E1))*2.0;
//                 tmp[1] = mu*(sin(M_PI/12.0)*exp(x)*(1.0/2.0)-(M_PI*M_PI)*cos(M_PI*y)*cos(M_PI*z)*(exp(x)*(1.0/1.0E1)-1.0/1.0E1)*(1.0/2.0))*2.0;
//                 tmp[2] = mu*(exp(x)*(cos(M_PI/12.0)-1.0)*(1.0/2.0)+(M_PI*M_PI)*sin(M_PI*y)*sin(M_PI*z)*(exp(x)*(1.0/1.0E1)-1.0/1.0E1)*(1.0/2.0))*2.0;
//                 grads[6] = tmp;

//                 tmp[0] = 0;
//                 tmp[1] = 0;
//                 tmp[2] = 0;
//                 grads[7] = tmp;

//                 tmp[0] = lmbda*(exp(x)*(cos(M_PI/12.0)-1.0)*2.0-exp(x)*sin(M_PI*y)*sin(M_PI*z)*(1.0/1.0E1))+mu*exp(x)*(cos(M_PI/12.0)-1.0)*2.0;
//                 tmp[1] = M_PI*lmbda*exp(x)*cos(M_PI*y)*sin(M_PI*z)*(-1.0/1.0E1);
//                 tmp[2] = M_PI*lmbda*exp(x)*cos(M_PI*z)*sin(M_PI*y)*(-1.0/1.0E1);
//                 grads[8] = tmp;

//                 // The gradient for the rest is meaningless
//                 tmp[0] = 0.0;
//                 tmp[1] = 0.0;
//                 tmp[2] = 0.0;
//                 for (int k=dim*dim;k<total_dim;++k)
//                     grads[k] = tmp;

//                 break;
//             default:
//             Assert(false, ExcNotImplemented());
//         }
//     }
}

#endif //ELASTICITY_MFEDD_DATA_H
