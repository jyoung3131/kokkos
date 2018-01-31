#include<cilk/cilk.h>

/*--------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------*/
/* Parallel patterns for Kokkos::CilkPlus with RangePolicy */

namespace Kokkos {
namespace Impl {

template< class FunctorType , class ... Traits >
class ParallelFor< FunctorType ,
                   Kokkos::RangePolicy< Traits ... > ,
                   Kokkos::CilkPlus
                 >
{
private:

  typedef Kokkos::RangePolicy< Traits ... > Policy ;

  const FunctorType m_functor ;
  const Policy      m_policy ;

  template< class TagType >
  typename std::enable_if< std::is_same< TagType , void >::value >::type
  exec() const
    {
      const typename Policy::member_type e = m_policy.end();
      cilk_for ( typename Policy::member_type i = m_policy.begin() ; i < e ; ++i ) {
        m_functor( i );
      }
    }

  template< class TagType >
  typename std::enable_if< ! std::is_same< TagType , void >::value >::type
  exec() const
    {
      const TagType t{} ;
      const typename Policy::member_type e = m_policy.end();
      cilk_for ( typename Policy::member_type i = m_policy.begin() ; i < e ; ++i ) {
        m_functor( t , i );
      }
    }

public:

  inline
  void execute() const
    { this-> template exec< typename Policy::work_tag >(); }

  inline
  ParallelFor( const FunctorType & arg_functor
             , const Policy      & arg_policy )
    : m_functor( arg_functor )
    , m_policy(  arg_policy )
    {}
};

}
}

