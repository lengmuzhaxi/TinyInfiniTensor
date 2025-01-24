#include "core/allocator.h"
#include <utility>

namespace infini
{
    Allocator::Allocator(Runtime runtime) : runtime(runtime)
    {
        used = 0;
        peak = 0;
        ptr = nullptr;

        // 'alignment' defaults to sizeof(uint64_t), because it is the length of
        // the longest data type currently supported by the DataType field of
        // the tensor
        alignment = sizeof(uint64_t);
    }

    Allocator::~Allocator()
    {
        if (this->ptr != nullptr)
        {
            runtime->dealloc(this->ptr);
        }
    }

    size_t Allocator::alloc(size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        // pad the size to the multiple of alignment
        size = this->getAlignedSize(size);

        // =================================== 作业 ===================================
        // TODO: 设计一个算法来分配内存，返回起始地址偏移量
        // =================================== 作业 ===================================
        for (auto it = free_blocks.begin(); it != free_blocks.end(); it++) {
          if (it->second >= size) {
            size_t addr = it->first;
            size_t addr_size = it->second - size;
            free_blocks.erase(it);
            if (addr_size > 0) {
              free_blocks[addr + size] = addr_size;
            }
            this->used += size;
            this->peak = std::max(this->used, this->peak);
            return addr;
          }
        }
        return 0;
    }

    void Allocator::free(size_t addr, size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        size = getAlignedSize(size);

        // =================================== 作业 ===================================
        // TODO: 设计一个算法来回收内存
        // =================================== 作业 ===================================
        auto next = free_blocks.upper_bound(addr);
    auto prev = (next == free_blocks.begin()) ? free_blocks.end() : std::prev(next);

    // 尝试合并前面的块
    if (prev != free_blocks.end() && prev->first + prev->second == addr) {
        addr = prev->first;
        size += prev->second;
        free_blocks.erase(prev);
    }

    // 尝试合并后面的块
    if (next != free_blocks.end() && addr + size == next->first) {
        size += next->second;
        free_blocks.erase(next);
    }

    // 插入合并后的块
    free_blocks[addr] = size;

    // 更新内存使用情况
    this->used -= size;
    }

    void *Allocator::getPtr()
    {
        if (this->ptr == nullptr)
        {
            this->ptr = runtime->alloc(this->peak);
            printf("Allocator really alloc: %p %lu bytes\n", this->ptr, peak);
        }
        return this->ptr;
    }

    size_t Allocator::getAlignedSize(size_t size)
    {
        return ((size - 1) / this->alignment + 1) * this->alignment;
    }

    void Allocator::info()
    {
        std::cout << "Used memory: " << this->used
                  << ", peak memory: " << this->peak << std::endl;
    }
}
