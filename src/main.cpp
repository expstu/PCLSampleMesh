#include <iostream>

#include <pcl/features/moment_of_inertia_estimation.h>
#include <pcl/common/common_headers.h>
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/time.h>   // TicToc
#include <pcl/octree/octree_pointcloud.h>
#include <pcl/PolygonMesh.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>

#include <Eigen/Geometry>

#include <vtkTriangle.h>
#include <vtkTriangleFilter.h>
#include <vtkPolyDataMapper.h>

//convenient typedefs
typedef pcl::PointXYZRGB PointT;
typedef pcl::PointCloud<PointT> PointCloud;

pcl::visualization::PCLVisualizer *viewer;
boost::mt19937 engine;
boost::uniform_01<boost::mt19937&> u01(engine);

inline double
uniform_deviate(int seed) {
	double ran = seed * (1.0 / (RAND_MAX + 1.0));
	return ran;
}

inline void
randomPointTriangle(float a1, float a2, float a3, float b1, float b2, float b3, float c1, float c2, float c3,
	Eigen::Vector4f& p)
{
	float r1 = static_cast<float> (uniform_deviate(rand()));
	float r2 = static_cast<float> (uniform_deviate(rand()));
	float r1sqr = std::sqrt(r1);
	float OneMinR1Sqr = (1 - r1sqr);
	float OneMinR2 = (1 - r2);
	a1 *= OneMinR1Sqr;
	a2 *= OneMinR1Sqr;
	a3 *= OneMinR1Sqr;
	b1 *= OneMinR2;
	b2 *= OneMinR2;
	b3 *= OneMinR2;
	c1 = r1sqr * (r2 * c1 + b1) + a1;
	c2 = r1sqr * (r2 * c2 + b2) + a2;
	c3 = r1sqr * (r2 * c3 + b3) + a3;
	p[0] = c1;
	p[1] = c2;
	p[2] = c3;
	p[3] = 0;
}

inline void
randPSurface(vtkPolyData * polydata, std::vector<double> * cumulativeAreas, double totalArea,
	Eigen::Vector4f& p, bool calcNormal, Eigen::Vector3f& n)
{
	double r = u01() * totalArea;

	std::vector<double>::iterator low = std::lower_bound(cumulativeAreas->begin(), cumulativeAreas->end(), r);
	vtkIdType el = vtkIdType(low - cumulativeAreas->begin());

	double A[3], B[3], C[3];
	vtkIdType npts = 0;
	vtkIdType *ptIds = NULL;
	polydata->GetCellPoints(el, npts, ptIds);
	polydata->GetPoint(ptIds[0], A);
	polydata->GetPoint(ptIds[1], B);
	polydata->GetPoint(ptIds[2], C);
	if (calcNormal)
	{
		// OBJ: Vertices are stored in a counter-clockwise order by default
		Eigen::Vector3f v1 = Eigen::Vector3f(A[0], A[1], A[2]) - Eigen::Vector3f(C[0], C[1], C[2]);
		Eigen::Vector3f v2 = Eigen::Vector3f(B[0], B[1], B[2]) - Eigen::Vector3f(C[0], C[1], C[2]);
		n = v1.cross(v2);
		n.normalize();
	}
	randomPointTriangle(float(A[0]), float(A[1]), float(A[2]),
		float(B[0]), float(B[1]), float(B[2]),
		float(C[0]), float(C[1]), float(C[2]), p);
}

void
uniform_sampling(vtkSmartPointer<vtkPolyData> polydata, size_t n_samples, bool calc_normal,
	pcl::PointCloud<pcl::PointNormal> & cloud_out)
{
	polydata->BuildCells();
	vtkSmartPointer<vtkCellArray> cells = polydata->GetPolys();

	double p1[3], p2[3], p3[3];
	double totalArea = 0;
	std::vector<double> cumulativeAreas(cells->GetNumberOfCells(), 0);
	size_t i = 0;
	vtkIdType npts = 0, *ptIds = NULL;
	for (cells->InitTraversal(); cells->GetNextCell(npts, ptIds); i++)
	{
		polydata->GetPoint(ptIds[0], p1);
		polydata->GetPoint(ptIds[1], p2);
		polydata->GetPoint(ptIds[2], p3);
		totalArea += vtkTriangle::TriangleArea(p1, p2, p3);
		cumulativeAreas[i] = totalArea;
	}

	cloud_out.points.resize(n_samples);
	cloud_out.width = static_cast<pcl::uint32_t> (n_samples);
	cloud_out.height = 1;

	for (i = 0; i < n_samples; i++)
	{
		Eigen::Vector4f p;
		Eigen::Vector3f n;
		randPSurface(polydata, &cumulativeAreas, totalArea, p, calc_normal, n);
		cloud_out.points[i].x = p[0];
		cloud_out.points[i].y = p[1];
		cloud_out.points[i].z = p[2];
		if (calc_normal)
		{
			cloud_out.points[i].normal_x = n[0];
			cloud_out.points[i].normal_y = n[1];
			cloud_out.points[i].normal_z = n[2];
		}
	}
}

template<typename PointType>
void getAABB(pcl::PointCloud<PointType>& cloud, PointType& minP, PointType& maxP) {
	minP = maxP = cloud[0];
	for (int i = 1; i < cloud.size(); i++) {
		minP.x = std::min(cloud[i].x, minP.x);
		minP.y = std::min(cloud[i].y, minP.y);
		minP.z = std::min(cloud[i].z, minP.z);
		maxP.x = std::max(cloud[i].x, maxP.x);
		maxP.y = std::max(cloud[i].y, maxP.y);
		maxP.z = std::max(cloud[i].z, maxP.z);
	}
}

const float baseHeight = 15;
const float sensorWidth = 11.264f;

int main(int argc, char** argv) {
	pcl::console::TicToc time;

	//////////////////////////////////////////////////////////////////////////
	// Load model
	time.tic();
	pcl::PolygonMesh polygonMesh;
	pcl::io::loadPolygonFileSTL("model.STL", polygonMesh);
	std::cout << time.toc() / 1000.0 << 's' << std::endl;

	pcl::PointCloud<pcl::PointXYZ>::Ptr meshCloud(new pcl::PointCloud<pcl::PointXYZ>);
	fromPCLPointCloud2(polygonMesh.cloud, *meshCloud);
	std::cout << "Mesh points: " << meshCloud->size() << std::endl;

	// Transform
	time.tic();
	pcl::PointXYZ minP2, maxP2;
	getAABB(*meshCloud, minP2, maxP2);
	Eigen::Affine3f a;
	a = Eigen::Translation3f(-(minP2.x + maxP2.x) / 2, -(minP2.y + maxP2.y) / 2, baseHeight);
	pcl::transformPointCloud(*meshCloud, *meshCloud, a);
	toPCLPointCloud2(*meshCloud, polygonMesh.cloud);
	getAABB(*meshCloud, minP2, maxP2);
	PCL_INFO("%f,%f,%f,%f,%f,%f time:%fs\n",
		minP2.x, maxP2.x, minP2.y, maxP2.y, minP2.z, maxP2.z, time.toc() / 1000.0);

	vtkSmartPointer<vtkPolyData> polydata1 = vtkSmartPointer<vtkPolyData>::New();
	pcl::io::mesh2vtk(polygonMesh, polydata1);

	vtkSmartPointer<vtkTriangleFilter> triangleFilter = vtkSmartPointer<vtkTriangleFilter>::New();
	triangleFilter->SetInputData(polydata1);

	vtkSmartPointer<vtkPolyDataMapper> triangleMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
	triangleMapper->SetInputConnection(triangleFilter->GetOutputPort());
	triangleMapper->Update();
	polydata1 = triangleMapper->GetInput();

	// Sampling
	time.tic();
	int SAMPLE_POINTS_ = 10000000;
	pcl::PointCloud<pcl::PointNormal>::Ptr sampledCloud(new pcl::PointCloud<pcl::PointNormal>);
	uniform_sampling(polydata1, SAMPLE_POINTS_, true,*sampledCloud);
	std::cout << "Sampled cloud size:" << sampledCloud->size() << " " << time.toc() / 1000. << "s" << std::endl;

	// Get top layer
	time.tic();
	pcl::PointCloud<pcl::PointNormal>::Ptr topSampledCloud(new pcl::PointCloud<pcl::PointNormal>);
	for (int i = 0; i < sampledCloud->size() ; i++) {
		if (sampledCloud->points[i].normal_z >= 0) {
			topSampledCloud->push_back(sampledCloud->points[i]);
		}
	}
	std::cout << "Cut cloud size:" << topSampledCloud->size() << " " << time.toc() / 1000. << "s" << std::endl;

	// Clip model
	time.tic();
	pcl::PointNormal minP, maxP;
	getAABB(*topSampledCloud, minP, maxP);

	pcl::PointCloud<pcl::PointNormal>::Ptr modelCloudClipped(new pcl::PointCloud<pcl::PointNormal>);
	pcl::PointCloud<pcl::PointNormal>::Ptr tempCloud(new pcl::PointCloud<pcl::PointNormal>);

	std::cout << "clipper " << std::endl;
	pcl::PassThrough<pcl::PointNormal> clipper;
	clipper.setInputCloud(topSampledCloud);
	clipper.setFilterFieldName("x");
	clipper.setFilterLimits(minP.x + sensorWidth, maxP.x - sensorWidth);
	clipper.setFilterLimitsNegative(true);
	clipper.filter(*tempCloud);

	clipper.setFilterLimitsNegative(false);
	clipper.filter(*modelCloudClipped);

	clipper.setInputCloud(modelCloudClipped);
	clipper.setFilterFieldName("y");
	clipper.setFilterLimits(minP.y + sensorWidth, maxP.y - sensorWidth);
	clipper.setFilterLimitsNegative(true);
	clipper.filter(*modelCloudClipped);

	*modelCloudClipped += *tempCloud;
	std::cout << time.toc() / 1000.0 << 's' << std::endl;
	std::cout << "Clipped cloud size: " << modelCloudClipped->size() << std::endl;

	// Down sample
	time.tic();
	std::cout << "Filter in: ";
	pcl::PointCloud<pcl::PointNormal>::Ptr modelCloudFiltered(new pcl::PointCloud<pcl::PointNormal>);
	pcl::VoxelGrid<pcl::PointNormal> sor;
	sor.setInputCloud(modelCloudClipped);
	sor.setLeafSize(0.1f, 0.1f, 0.03f);
	sor.filter(*modelCloudFiltered);
	std::cout << time.toc() / 1000.0 << 's' << std::endl;
	std::cout << "Filtered cloud size: " << modelCloudFiltered->size() << std::endl;

	// Output
	pcl::io::savePCDFileBinary("model_cloud.pcd", *modelCloudClipped);
	pcl::io::savePCDFileBinary("model_cloud_filtered.pcd", *modelCloudFiltered);

	//////////////////////////////////////////////////////////////////////////
	// show

/*
	viewer = new pcl::visualization::PCLVisualizer(argc, argv, "Sample mesh test");

	//viewer->addPolygonMesh(polygonMesh, "mesh");
	//pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> handler(meshCloud, 0, 255, 0);
	//viewer->addPointCloud(meshCloud, handler2, "mesh_cloud");

	//viewer->addModelFromPolyData(polydata1, "mesh1");

	//pcl::visualization::PointCloudColorHandlerCustom<pcl::PointNormal> handler1(cloud_1, 255, 255, 0);
	//viewer->addPointCloud<pcl::PointNormal>(cloud_1, handler1, "cloud1");
	//viewer->addPointCloudNormals<pcl::PointNormal>(cloud_1, 1, 0.02f, "cloud_normals");


	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointNormal> handler2(modelCloudFiltered, 0, 0, 255);
	viewer->addPointCloud<pcl::PointNormal>(modelCloudFiltered, handler2, "cloud2");
	viewer->addPointCloudNormals<pcl::PointNormal>(modelCloudFiltered, 1, 0.03f, "cloud_2_normals");

	// bounding box
	//viewer->addCube(minP2.x, maxP2.x, minP2.y, maxP2.y, minP2.z, maxP2.z, 1.0, 0.0, 0.0, "meshAABB");
	//viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION,
	//	pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME, "meshAABB");
	//viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3.0, "meshAABB");

	viewer->addCoordinateSystem(1.0);
	viewer->initCameraParameters();
	viewer->setCameraPosition(0.013336, -1.73476, 144.206, -0.999603, 0.0281814, 0.000434471);
	viewer->setSize(1366, 768);

	while (!viewer->wasStopped()) {
		viewer->spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}*/

	return 0;
}