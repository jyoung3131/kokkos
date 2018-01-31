#ifndef KOKKOS_CILKPLUS_MDRANGE_HPP
#define KOKKOS_CILKPLUS_MDRANGE_HPP

#include<cilk/cilk.h>

namespace Kokkos {
namespace Impl {


template< class FunctorType , class ... Traits >
class ParallelFor< FunctorType ,
                   Kokkos::MDRangePolicy< Traits ... > ,
                   Kokkos::Experimental::CilkPlus
                 >
{
private:

  typedef Kokkos::MDRangePolicy< Traits ... > MDRangePolicy ;
  typedef typename MDRangePolicy::impl_range_policy Policy ;

  typedef typename Kokkos::Impl::HostIterateTile< MDRangePolicy, FunctorType, typename MDRangePolicy::work_tag, void > iterate_type;

  const FunctorType   m_functor ;
  const MDRangePolicy m_mdr_policy ;
  const Policy        m_policy ;

  void
  exec() const
    {
      const typename Policy::member_type e = m_policy.end();
      cilk_for ( typename Policy::member_type i = m_policy.begin() ; i < e ; ++i ) {
        iterate_type( m_mdr_policy, m_functor )( i );
      }
    }

public:

  inline
  void execute() const
    { this->exec(); }

  inline
  ParallelFor( const FunctorType   & arg_functor
             , const MDRangePolicy & arg_policy )
    : m_functor( arg_functor )
    , m_mdr_policy(  arg_policy )
    , m_policy( Policy(0, m_mdr_policy.m_num_tiles).set_chunk_size(1) )
    {}
};

template< class FunctorType , class ReducerType , class ... Traits >
class ParallelReduce< FunctorType
                    , Kokkos::MDRangePolicy< Traits ... >
                    , ReducerType
                    , Kokkos::Experimental::CilkPlus
                    >
{
private:

  typedef Kokkos::MDRangePolicy< Traits ... > MDRangePolicy ;
  typedef typename MDRangePolicy::impl_range_policy Policy ;

  typedef typename MDRangePolicy::work_tag                                  WorkTag ;

  typedef Kokkos::Impl::if_c< std::is_same<InvalidType,ReducerType>::value, FunctorType, ReducerType> ReducerConditional;
  typedef typename ReducerConditional::type ReducerTypeFwd;
  typedef typename Kokkos::Impl::if_c< std::is_same<InvalidType,ReducerType>::value, WorkTag, void>::type WorkTagFwd;

  typedef typename ReducerTypeFwd::value_type ValueType;

  typedef FunctorAnalysis< FunctorPatternInterface::REDUCE , Policy , FunctorType > Analysis ;

  typedef Kokkos::Impl::FunctorValueInit<   ReducerTypeFwd , WorkTagFwd >  ValueInit ;

  typedef typename Analysis::pointer_type    pointer_type ;
  typedef typename Analysis::reference_type  reference_type ;


  using iterate_type = typename Kokkos::Impl::HostIterateTile< MDRangePolicy
                                                                           , FunctorType
                                                                           , WorkTag
                                                                           , ValueType
                                                                           >;


  const FunctorType   m_functor ;
  const MDRangePolicy m_mdr_policy ;
  const Policy        m_policy ;
  const ReducerType   m_reducer ;
  const pointer_type  m_result_ptr ;

  inline
  void
  exec( reference_type update ) const
    {
      static_assert(std::is_same<ReducerType,Kokkos::Experimental::Sum<typename Analysis::value_type>>::value ||
                    std::is_same<ReducerType,Kokkos::InvalidType>::value ,"Only Sum Reductions are supported by the CilkPlus execution space");
      const typename Policy::member_type e = m_policy.end();
      cilk::reducer< cilk::op_add<typename Analysis::value_type> > parallel_sum(0);
      cilk_for ( typename Policy::member_type i = m_policy.begin() ; i < e ; ++i ) {
        typename Analysis::value_type lupdate = 0;
        iterate_type( m_mdr_policy, m_functor, lupdate )( i );
        *parallel_sum += lupdate;
      }
      update = parallel_sum.get_value();
    }

public:

  inline
  void execute() const
    {
      const size_t pool_reduce_size =
        Analysis::value_size( ReducerConditional::select(m_functor , m_reducer) );
      const size_t team_reduce_size  = 0 ; // Never shrinks
      const size_t team_shared_size  = 0 ; // Never shrinks
      const size_t thread_local_size = 0 ; // Never shrinks

      serial_resize_thread_team_data( pool_reduce_size
                                    , team_reduce_size
                                    , team_shared_size
                                    , thread_local_size );

      HostThreadTeamData & data = *serial_get_thread_team_data();

      pointer_type ptr =
        m_result_ptr ? m_result_ptr : pointer_type(data.pool_reduce_local());

      reference_type update =
        ValueInit::init(  ReducerConditional::select(m_functor , m_reducer) , ptr );

      this-> exec( update );

      Kokkos::Impl::FunctorFinal< ReducerTypeFwd , WorkTagFwd >::
        final(  ReducerConditional::select(m_functor , m_reducer) , ptr );
    }

  template< class HostViewType >
  ParallelReduce( const FunctorType  & arg_functor ,
                  const MDRangePolicy       & arg_policy ,
                  const HostViewType & arg_result_view ,
                  typename std::enable_if<
                               Kokkos::is_view< HostViewType >::value &&
                              !Kokkos::is_reducer_type<ReducerType>::value
                  ,void*>::type = NULL)
    : m_functor( arg_functor )
    , m_mdr_policy( arg_policy )
    , m_policy( Policy(0, m_mdr_policy.m_num_tiles).set_chunk_size(1) )
    , m_reducer( InvalidType() )
    , m_result_ptr( arg_result_view.data() )
    {
      static_assert( Kokkos::is_view< HostViewType >::value
        , "Kokkos::Experimental::CilkPlus reduce result must be a View" );

      static_assert( std::is_same< typename HostViewType::memory_space , HostSpace >::value
        , "Kokkos::Experimental::CilkPlus reduce result must be a View in HostSpace" );
    }

  inline
  ParallelReduce( const FunctorType & arg_functor
                , MDRangePolicy       arg_policy
                , const ReducerType& reducer )
    : m_functor( arg_functor )
    , m_mdr_policy(  arg_policy )
    , m_policy( Policy(0, m_mdr_policy.m_num_tiles).set_chunk_size(1) )
    , m_reducer( reducer )
    , m_result_ptr(  reducer.view().data() )
    {
    }
};
}
}

#endif
