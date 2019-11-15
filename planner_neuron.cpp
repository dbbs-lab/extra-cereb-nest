#include "planner_neuron.h"

// C++ includes:
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <limits>
#include <fstream>

#include "poisson_generator.h"

// Includes from libnestutil:
#include "numerics.h"

// Includes from nestkernel:
#include "event_delivery_manager_impl.h"
#include "exceptions.h"
#include "kernel_manager.h"

// Includes from sli:
#include "dict.h"
#include "dictutils.h"
#include "doubledatum.h"
#include "integerdatum.h"


/* ----------------------------------------------------------------
 * Default constructors defining default parameters and state
 * ---------------------------------------------------------------- */

mynest::planner_neuron::Parameters_::Parameters_()
  : trial_length_( 1000 )
  , target_( 0.0 )
  , prism_deviation_( 0.0 )
  , baseline_rate_( 10.0 )
  , gain_rate_( 1.0 )
{
}

/* ----------------------------------------------------------------
 * Parameter and state extractions and manipulation functions
 * ---------------------------------------------------------------- */

void
mynest::planner_neuron::Parameters_::get( DictionaryDatum& d ) const
{
  def< long >( d, mynames::trial_length, trial_length_ );
  def< double >( d, mynames::target, target_ );
  def< double >( d, mynames::prism_deviation, prism_deviation_ );
  def< double >( d, mynames::baseline_rate, baseline_rate_ );
  def< double >( d, mynames::gain_rate, gain_rate_ );
}

void
mynest::planner_neuron::Parameters_::set( const DictionaryDatum& d )
{
  updateValue< long >( d, mynames::trial_length, trial_length_ );
  if ( trial_length_ <= 0 )
  {
    throw nest::BadProperty( "The trial length cannot be zero or negative." );
  }

  updateValue< double >( d, mynames::target, target_ );
  updateValue< double >( d, mynames::prism_deviation, prism_deviation_ );
  updateValue< double >( d, mynames::baseline_rate, baseline_rate_ );
  updateValue< double >( d, mynames::gain_rate, gain_rate_ );
}

/* ----------------------------------------------------------------
 * Default and copy constructor for node, and destructor
 * ---------------------------------------------------------------- */

mynest::planner_neuron::planner_neuron()
  : Archiving_Node()
  , P_()
  , V_()
{
}

mynest::planner_neuron::planner_neuron( const planner_neuron& n )
  : Archiving_Node( n )
  , P_( n.P_ )
{
}

mynest::planner_neuron::~planner_neuron()
{
}

/* ----------------------------------------------------------------
 * Node initialization functions
 * ---------------------------------------------------------------- */

void
mynest::planner_neuron::init_state_( const Node& proto )
{
}


void
mynest::planner_neuron::init_buffers_()
{
  Archiving_Node::clear_history();
}

void
mynest::planner_neuron::calibrate()
{
  double rate = P_.baseline_rate_ + P_.gain_rate_ * (P_.target_ + P_.prism_deviation_);
  rate =  std::max(0.0, rate);
  V_.rate_ = rate;

  double time_res = nest::Time::get_resolution().get_ms();  // 0.1
  long ticks = (double)P_.trial_length_ / time_res;

  librandom::RngPtr rng = nest::kernel().rng_manager.get_rng( get_thread() );

  V_.poisson_dev_.set_lambda( time_res * rate * 1e-3 );

  for (long t = 0; t < ticks; t++ )
  {
    long n_spikes = V_.poisson_dev_.ldev( rng );

    if ( n_spikes > 0 ) // we must not send events with multiplicity 0
    {
      B_.spikes_[t] = n_spikes;
    }
  }
}


void
mynest::planner_neuron::update( nest::Time const& origin, const long from, const long to )
{
  assert( to >= 0 );
  assert( static_cast<nest::delay>(from) < nest::kernel().connection_manager.get_min_delay() );
  assert( from < to );

  double time_res = nest::Time::get_resolution().get_ms();  // 0.1
  long trial_ticks = (double)P_.trial_length_ / time_res;

  for ( long lag = from; lag < to; ++lag )
  {
    long t = origin.get_steps() + lag;
    int n_spikes = B_.spikes_[t % trial_ticks];

    if ( n_spikes > 0 )
    {
      nest::SpikeEvent se;
      se.set_multiplicity( n_spikes );
      nest::kernel().event_delivery_manager.send( *this, se, lag );

      // set the spike times, respecting the multiplicity
      for ( int i = 0; i < n_spikes; i++ )
      {
        set_spiketime( nest::Time::step( t ) );
      }
    }
  }
}

void
mynest::planner_neuron::handle( nest::SpikeEvent& e )
{
}