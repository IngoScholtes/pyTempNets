using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

using NETVisualizer;
using OpenTK;
using System.Collections.Concurrent;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;

namespace TemporalNetworks.TempNetLayout
{
    public enum ScalingMethod { Linear = 0, Sqrt = 1, Log = 2 }

    public class ForceBasedLayout : LayoutProvider
    {
        private Dictionary<string, Vector3> _vertexPositions;

        private ConcurrentBag<string> _dirtyVertices;

        private Random r;

        private TemporalNetwork temp_net;
        private TemporalNetwork temp_net_null;

        public ScalingMethod ScalingMethod = ScalingMethod.Linear;

        public int Iterations
        {
            get;
            set;
        }        

        public double BPInfluence
        {
            get;
            set;
        }

        public double WeightInfluence
        {
            get;
            set;
        }

        public double EdgeInfluence
        {
            get;
            set; 
        }

        public double RepulsionFactor
        {
            get;
            set;
        }

        public double AreaMultiplicator
        {
            get;
            set;
        }

        public double SpeedDivisor
        {
            get;
            set;
        }

        public double Speed
        {
            get;
            set;
        }

        public double Gravity
        {
            get;
            set;
        }

        public ForceBasedLayout(int iterations)
        {
            Iterations = iterations;
            BPInfluence = 0d;
            WeightInfluence = 0d;
            RepulsionFactor = 1d;
            EdgeInfluence = 1d;          
            AreaMultiplicator = 10000d;
            Speed = 1d;
            SpeedDivisor = 800d;
            Gravity = 10d;

            r = new Random();

            _vertexPositions = new Dictionary<string, Vector3>();            
        }

        public override void Init(double width, double height, IRenderableNet network)
        {
            base.Init(width, height, network);

            temp_net = (network as RenderableTempNet).Network;
            //temp_net_null = TemporalNetworkEnsemble.ShuffleEdges(temp_net);

            CreateRandomState();                               
        }

        /// <summary>
        /// Creates a random initial state where vertices a randomly placed in the given layout area
        /// </summary>
        private void CreateRandomState()
        {
            _dirtyVertices = new ConcurrentBag<string>();
            foreach (string v in base.Network.GetVertexArray())
            {
                _vertexPositions[v] = new Vector3((float)(r.NextDouble() * base.Width - base.Width/2d), (float)(r.NextDouble() * base.Height-base.Height/2d), 1f);
                _dirtyVertices.Add(v);
            }
        }

        /// <summary>
        /// Returns the cumulative (Euclidean) distance across all two-paths in the temporal network.
        /// This is what this force-based layout algorithm tries to minimize
        /// </summary>
        public double CumulativeTwoPathDistance
        {
            get
            {
                double CumulativeEdgeDistance = 0d;
                foreach (var two_path in temp_net.SecondOrderAggregateNetwork.Edges)
                {
                    string source_node = two_path.Item1.Split(';')[0].Substring(1);
                    string target_node = two_path.Item2.Split(';')[1].Trim(')');
                    CumulativeEdgeDistance += temp_net.SecondOrderAggregateNetwork[two_path] * (_vertexPositions[source_node] - _vertexPositions[target_node]).Length;
                }
                return CumulativeEdgeDistance;
            }
        }

        /// <summary>
        /// Returns the cumulative (Euclidean) distance across all edges in the aggregate network.
        /// This is what the Fruchterman-Reingold layout of a static network essentially tries to minimize 
        /// </summary>
        public double CumulativeEdgeDistance
        {
            get{
                double CumulativeEdgeDistance = 0d;
                foreach (var edge in temp_net.AggregateNetwork.Edges)
                    CumulativeEdgeDistance += temp_net.AggregateNetwork[edge] * (_vertexPositions[edge.Item1] - _vertexPositions[edge.Item2]).Length;
                return CumulativeEdgeDistance;
            }
        }


        public override void DoLayout()
        {
            double area = Width * Height;

            // The displacement calculated for each vertex in each step
            var disp = new ConcurrentDictionary<string, Vector3>(System.Environment.ProcessorCount, (int)Network.GetVertexCount());

            // Some area dependent parameters
            double maxDist =  (Math.Sqrt(area * AreaMultiplicator) / 10d);
            double k = Math.Sqrt(AreaMultiplicator * area)/ (1d + Network.GetVertexCount());

            var vertices = Network.GetVertexArray();
            var edges = Network.GetEdgeArray();

            foreach(string v in vertices)
                    disp[v] = new Vector3(0f, 0f, 1f);

            for (int i = 0; i < Iterations; i++)
            {
                // parallely compute repulsive forces of nodes to every new node
//#if DEBUG
                foreach (var v in vertices)
//#else   
 //               Parallel.ForEach(_dirtyVertices, v =>
//#endif
                {                    
                    foreach (string u in vertices)
                    {
                        if (v != u)
                        {
                            // Compute repulsive force

                            Vector3 delta = _vertexPositions[v] - _vertexPositions[u];
                            disp[v] = disp[v] + Vector3.Multiply(Vector3.Divide(delta, delta.Length), (float)(RepulsionFactor * k * k / delta.Length));

                            // Compute attractive force
                            if (_dirtyVertices.Contains(v))
                                disp[v] = disp[v] - Vector3.Multiply(Vector3.Divide(delta, delta.Length), (float) attraction(delta.Length, k, v, u));
                            if (_dirtyVertices.Contains(u))
                                disp[u] = disp[u] + Vector3.Multiply(Vector3.Divide(delta, delta.Length), (float) attraction(delta.Length, k, v, u));
                        }
                    }
                }
//#if !DEBUG
 //               );
//#endif
                foreach (string v in _dirtyVertices)
                {
                    double dist = disp[v].Length;
                    double new_x = disp[v].X - (0.01d * k * Gravity * _vertexPositions[v].X);
                    double new_y = disp[v].Y - (0.01d * k * Gravity * _vertexPositions[v].Y);
                    disp[v] = new Vector3((float) new_x, (float) new_y, 1f);
                }
//#if DEBUG
                foreach (var v in _dirtyVertices)
//#else
 //               Parallel.ForEach(_dirtyVertices, v =>
//#endif
                    {
                        Vector3 vPos = _vertexPositions[v] + Vector3.Multiply(Vector3.Divide(disp[v], disp[v].Length), 
                            (float)Math.Min(disp[v].Length, maxDist * (Speed / SpeedDivisor)));

                        // We skip the limitation to a certain frame, since we can still pan and zoom ... 
                        //vPos.X = (float)Math.Min(Width - 10, Math.Max(10, vPos.X));
                        //vPos.Y = (float)Math.Min(Height - 10, Math.Max(10, vPos.Y));

                        _vertexPositions[v] = vPos;
                        disp[v] = new Vector3(0f, 0f, 1f);
                    }
//#if !DEBUG
    //            );
//#endif
            }
            _dirtyVertices = new ConcurrentBag<string>();
            Console.WriteLine("Cumulative edge distance = {0:0.000}", CumulativeEdgeDistance);
            //Console.WriteLine("Cumulative two-path distance = {0:0.000}", CumulativeTwoPathDistance);
        }     


        /// <summary>
        /// Computes the attractive force between a pair of nodes v and w
        /// </summary>
        /// <param name="distance"></param>
        /// <param name="k"></param>
        /// <param name="v"></param>
        /// <param name="w"></param>
        /// <returns></returns>
        double attraction(double distance, double k, string v, string w)
        {
            // We compute three attractive forces: based on topology, weights and two-paths
            double weight_attraction = Math.Max(temp_net.AggregateNetwork.GetWeight(v, w), temp_net.AggregateNetwork.GetWeight(w, v));
            double edge_attraction = temp_net.AggregateNetwork.GetSuccessors(v).Contains(w) ? 1d : 0d;
            double bp_attraction = 0d;

            // Along all edges (v;x) -> (x;w) in the second-order network, add an attractive force between v and w
            foreach (string x in temp_net.AggregateNetwork.GetSuccessors(v))
            {
                var two_path = new Tuple<string, string>(string.Format("({0};{1})", v, x), string.Format("({0};{1})", x, w));
                if(BPInfluence>0d)
                    if (temp_net.SecondOrderAggregateNetwork.ContainsKey(two_path))
                        bp_attraction += (temp_net.SecondOrderAggregateNetwork[two_path] / temp_net.SecondOrderAggregateNetwork.GetCumulativeOutWeight(string.Format("({0};{1})", v, x)));
            }           
            
            // Scale the forces by the distance
            bp_attraction *= distance * distance / k;
            weight_attraction *= distance * distance / k;
            edge_attraction *= distance * distance / k;

            // Combine the three forces depending on  influence factors
            return (edge_attraction * EdgeInfluence + weight_attraction * WeightInfluence + bp_attraction * BPInfluence) * AreaMultiplicator;
        }

        /// <summary>
        /// Returns the position of a node v
        /// </summary>
        /// <param name="v">The node v for which the position is returned</param>
        /// <returns></returns>
        [MethodImpl(MethodImplOptions.Synchronized)]
        public override OpenTK.Vector3 GetPositionOfNode(string v)
        {
            return _vertexPositions[v];
        }       
    }
}
