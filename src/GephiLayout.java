import java.nio.file.*;
import java.nio.charset.StandardCharsets;
import java.lang.reflect.*;
import java.util.*;

/** Headless layout with the installed Gephi graphstore and ForceAtlas2. */
public class GephiLayout {
    static Object call(Object o,String name,Class<?>[] types,Object... args) throws Exception {
        Method m=o.getClass().getMethod(name,types);m.setAccessible(true);return m.invoke(o,args);
    }
    static Class<?> type(String name) throws Exception {return Class.forName(name);}
    public static void main(String[] args) throws Exception {
        Object model=type("org.gephi.graph.api.GraphModel$Factory").getMethod("newInstance").invoke(null);
        Object factory=call(model,"factory",new Class<?>[]{});
        Object graph=call(model,"getUndirectedGraph",new Class<?>[]{});
        Class<?> nodeType=type("org.gephi.graph.api.Node"),edgeType=type("org.gephi.graph.api.Edge");
        Map<String,Object> nodes=new LinkedHashMap<>();Random random=new Random(42);
        List<String> lines=Files.readAllLines(Paths.get(args[0]),StandardCharsets.UTF_8);
        for(int i=1;i<lines.size();i++) {
            String id=lines.get(i).split("\t")[0];Object n=call(factory,"newNode",new Class<?>[]{Object.class},id);
            call(n,"setX",new Class<?>[]{float.class},(random.nextFloat()-.5f)*100);
            call(n,"setY",new Class<?>[]{float.class},(random.nextFloat()-.5f)*100);
            call(n,"setSize",new Class<?>[]{float.class},2f);
            call(graph,"addNode",new Class<?>[]{nodeType},n);nodes.put(id,n);
        }
        lines=Files.readAllLines(Paths.get(args[1]),StandardCharsets.UTF_8);
        for(int i=1;i<lines.size();i++) {
            String[] s=lines.get(i).split("\t");Object e=call(factory,"newEdge",new Class<?>[]{nodeType,nodeType,int.class,double.class,boolean.class},nodes.get(s[0]),nodes.get(s[1]),0,Double.valueOf(s[2]),false);
            call(graph,"addEdge",new Class<?>[]{edgeType},e);
        }
        Object layout=type("org.gephi.layout.plugin.forceAtlas2.ForceAtlas2").getConstructors()[0].newInstance(new Object[]{null});
        call(layout,"setGraphModel",new Class<?>[]{type("org.gephi.graph.api.GraphModel")},model);
        call(layout,"resetPropertiesValues",new Class<?>[]{});
        call(layout,"setThreadsCount",new Class<?>[]{Integer.class},1);
        call(layout,"setScalingRatio",new Class<?>[]{Double.class},30.0);
        call(layout,"setGravity",new Class<?>[]{Double.class},1.0);
        call(layout,"setBarnesHutOptimize",new Class<?>[]{Boolean.class},false);
        call(layout,"setLinLogMode",new Class<?>[]{Boolean.class},false);
        call(layout,"setInitialisationSeed",new Class<?>[]{long.class},42L);
        call(layout,"initAlgo",new Class<?>[]{});
        for(int i=0;i<1200;i++)call(layout,"goAlgo",new Class<?>[]{});
        call(layout,"endAlgo",new Class<?>[]{});
        List<String> out=new ArrayList<>();out.add("feature,x,y");
        for(Map.Entry<String,Object> item:nodes.entrySet())out.add(item.getKey()+","+call(item.getValue(),"x",new Class<?>[]{})+","+call(item.getValue(),"y",new Class<?>[]{}));
        Files.write(Paths.get(args[2]),out,StandardCharsets.UTF_8);
        System.out.println("Gephi ForceAtlas2: "+nodes.size()+" nodes; 1200 iterations; seed 42; one thread");
    }
}
