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
	//float r = static_cast<double> (uniform_deviate(rand()) * totalArea);
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
		/*
		// every triangle should have at least 3 sample points
		Eigen::Vector4f p;
		Eigen::Vector3f n;
		pcl::PointNormal pclP;
		for (int i = 0; i < 3; i++) {
		randomPointTriangle(float(p1[0]), float(p1[1]), float(p1[2]),
		float(p2[0]), float(p2[1]), float(p2[2]),
		float(p3[0]), float(p3[1]), float(p3[2]), p);
		pclP.x = p[0];
		pclP.y = p[1];
		pclP.z = p[2];
		if (calc_normal)
		{
		// OBJ: Vertices are stored in a counter-clockwise order by default
		Eigen::Vector3f v1 = Eigen::Vector3f(p1[0], p1[1], p1[2]) - Eigen::Vector3f(p3[0], p3[1], p3[2]);
		Eigen::Vector3f v2 = Eigen::Vector3f(p2[0], p2[1], p2[2]) - Eigen::Vector3f(p3[0], p3[1], p3[2]);
		n = v1.cross(v2);
		n.normalize();
		pclP.normal_x = n[0];
		pclP.normal_y = n[1];
		pclP.normal_z = n[2];
		}
		cloud_out.points.push_back(pclP);
		}*/
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

int main(int argc, char** argv) {
	pcl::console::TicToc time;

	//////////////////////////////////////////////////////////////////////////
	// load model
	time.tic();
	pcl::PolygonMesh polygonMesh;
	pcl::io::loadPolygonFileSTL("cp2017.STL", polygonMesh);
	std::cout << time.toc() / 1000.0 << 's' << std::endl;

	pcl::PointCloud<pcl::PointXYZ>::Ptr meshCloud(new pcl::PointCloud<pcl::PointXYZ>);
	fromPCLPointCloud2(polygonMesh.cloud, *meshCloud);
	std::cout << "Mesh points: " << meshCloud->size() << std::endl;

	time.tic();
	pcl::PointXYZ minP2, maxP2;
	getAABB(*meshCloud, minP2, maxP2);
	// transform
	Eigen::Affine3f a;
	a = Eigen::AngleAxisf(180 * static_cast<float>(M_PI) / 180, Eigen::Vector3f::UnitY())*
		Eigen::Translation3f(-(minP2.x + maxP2.x) / 2, -(minP2.y + maxP2.y) / 2, -maxP2.z);
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

	time.tic();
	int SAMPLE_POINTS_ = 10000000;
	pcl::PointCloud<pcl::PointNormal>::Ptr cloud_1(new pcl::PointCloud<pcl::PointNormal>);
	uniform_sampling(polydata1, SAMPLE_POINTS_, true, *cloud_1);
	std::cout << "Sampled cloud size:" << cloud_1->size() << " " << time.toc() / 1000. << "s" << std::endl;

	/*
	// Voxelgrid
	pcl::VoxelGrid<pcl::PointNormal> grid_;
	grid_.setInputCloud(cloud_1);
	float leaf_size = 1.f;
	grid_.setLeafSize(leaf_size, leaf_size, leaf_size);

	time.tic();
	pcl::PointCloud<pcl::PointNormal>::Ptr voxel_cloud(new pcl::PointCloud<pcl::PointNormal>);
	grid_.filter(*voxel_cloud);
	std::cout << voxel_cloud->size() << std::endl;
	std::cout << "Filtered cloud size: " << time.toc() / 1000. << "s" << std::endl;*/

	//////////////////////////////////////////////////////////////////////////
	// show
	viewer = new pcl::visualization::PCLVisualizer(argc, argv, "Sample mesh test");

	//viewer->addPolygonMesh(polygonMesh, "mesh");
	//pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> handler(meshCloud, 0, 255, 0);
	//viewer->addPointCloud(meshCloud, handler2, "mesh_cloud");

	//viewer->addModelFromPolyData(polydata1, "mesh1");

	//pcl::visualization::PointCloudColorHandlerCustom<pcl::PointNormal> handler2(cloud_1, 255, 255, 0);
	//viewer->addPointCloud<pcl::PointNormal>(cloud_1, handler2, "cloud1");
	//viewer->addPointCloudNormals<pcl::PointNormal>(cloud_1, 1, 0.02f, "cloud_normals");

	//viewer->addPointCloud<pcl::PointNormal>(voxel_cloud);
	//viewer->addPointCloudNormals<pcl::PointNormal>(voxel_cloud, 1, 0.02f, "cloud_normals");
	//viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1.0, 0.0, 0.0, "cloud");

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
	}

	pcl::io::savePCDFileBinary("model_cloud.pcl", *cloud_1);

	return 0;
}