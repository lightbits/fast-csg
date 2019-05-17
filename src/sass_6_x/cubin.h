#include <stdio.h>
#include <assert.h>
#include <stdint.h>
#include <string.h>

struct elf64_hdr_t
{
    uint8_t magic[4];
    uint8_t fileClass;
    uint8_t encoding;
    uint8_t fileVersion;
    uint8_t padding[9];
    uint16_t type;
    uint16_t machine;
    uint32_t version;
    uint64_t entry;
    uint64_t phOffset;
    uint64_t shOffset;
    uint32_t flags;
    uint16_t ehSize;
    uint16_t phEntSize;
    uint16_t phNum;
    uint16_t shEntSize;
    uint16_t shNum;
    uint16_t shStrIndx;
};

struct elf64_prg_hdr_t
{
    uint32_t type;
    uint32_t flags;
    uint64_t offset;
    uint64_t vaddr;
    uint64_t paddr;
    uint64_t fileSize;
    uint64_t memSize;
    uint64_t align;
};

struct elf64_sec_hdr_t
{
    uint32_t name;
    uint32_t type;
    uint64_t flags;
    uint64_t addr;
    uint64_t offset;
    uint64_t size;
    uint32_t link;
    uint32_t info;
    uint64_t align;
    uint64_t entSize;
};

struct elf64_sym_ent_t
{
    uint32_t name;
    uint8_t  info;
    uint8_t  other;
    uint16_t shIndx;
    uint64_t value;
    uint64_t size;
};

struct cubin_function_t
{
    char *name;
    char *b;
    elf64_sec_hdr_t *h;
    elf64_sym_ent_t *e;

    #if 0
    uint64_t *instructions()           { return (uint64_t*)(b + h->offset); }
    int num_instructions()             { return (int)(h->size / sizeof(uint64_t)); }
    void set_num_instructions(int n)   { assert(n >= 0); h->size = n*sizeof(uint64_t); }
    #else
    // e->value is non-zero if the function is inlined, in which case it describe the
    // byte offset of the first instruction in the containing function's instructions.
    uint64_t *instructions()           { return (uint64_t*)(b + h->offset + e->value); }
    int num_instructions()             { return (int)(e->size/sizeof(uint64_t)); }
    void set_num_instructions(int n)
    {
        assert(n >= 0);
        assert(e->size == h->size && "The function appears to be an inline function. Changing the size of these is beyond the scope of this program.");
        e->size = ((uint64_t)n)*sizeof(uint64_t);
        h->size = ((uint64_t)n)*sizeof(uint64_t);
    }
    #endif

    uint8_t register_count()           { return (h->info & 0xff000000)>>24; }
    void set_register_count(uint8_t n) { h->info = (h->info & 0x00ffffff) | (n<<24); }
};

enum { cubin_max_prg_hdrs = 1024 };
enum { cubin_max_sec_hdrs = 1024 };
enum { cubin_max_functions = 1024 };
struct cubin_t
{
    int              sizeof_binary;
    char            *binary;
    elf64_prg_hdr_t *prg_hdrs[cubin_max_prg_hdrs];
    int              num_prg_hdrs;

    elf64_sec_hdr_t *sec_hdrs[cubin_max_sec_hdrs];
    int              num_sec_hdrs;

    cubin_function_t functions[cubin_max_functions];
    int              num_functions;

    cubin_function_t *get_function(const char *name)
    {
        for (int i = 0; i < num_functions; i++)
            if (strcmp(functions[i].name, name) == 0)
                return functions + i;
        return NULL;
    }
};

cubin_t read_cubin(const char *filename)
{
    {
        uint16_t x = 0xaabb;
        uint8_t *p = (uint8_t*)&x;
        assert(p[0] == 0xbb && "machine is not little (?) endian");
    }

    cubin_t cubin = {0};
    {
        FILE *f = fopen(filename, "rb");
        assert(f);
        fseek(f, 0, SEEK_END);
        long size = ftell(f);
        rewind(f);
        char *data = new char[size + 1];
        int ok = fread(data, 1, size, f);
        assert(ok);
        data[size] = 0;
        fclose(f);

        cubin.binary = data;
        cubin.sizeof_binary = size;
    }
    assert(cubin.binary);
    assert(cubin.sizeof_binary);

    elf64_hdr_t elf_hdr = *(elf64_hdr_t*)cubin.binary;
    assert(elf_hdr.fileClass == 2 && "assuming 64-bit ELF");
    assert((elf_hdr.flags & 0xff) == 60 && "assuming sm_60 architecture");
    assert(elf_hdr.flags & 0x400 && "assuming 64-bit addresses");
    assert(elf_hdr.phNum <= cubin_max_prg_hdrs);
    assert(elf_hdr.shNum <= cubin_max_sec_hdrs);

    // read program headers
    {
        char *b = cubin.binary + elf_hdr.phOffset;
        for (int i = 0; i < elf_hdr.phNum; i++)
        {
            cubin.prg_hdrs[cubin.num_prg_hdrs++] = (elf64_prg_hdr_t*)b;
            b += elf_hdr.phEntSize;
        }
    }

    // read section headers
    {
        char *b = cubin.binary + elf_hdr.shOffset;
        for (int i = 0; i < elf_hdr.shNum; i++)
        {
            cubin.sec_hdrs[cubin.num_sec_hdrs++] = (elf64_sec_hdr_t*)b;
            b += elf_hdr.shEntSize;
        }
    }


    // find section headers called strtab and shstrtab
    char *strtab = NULL;
    char *shstrtab = NULL;
    for (int i = 0; i < cubin.num_sec_hdrs; i++)
    {
        elf64_sec_hdr_t *sh = (elf64_sec_hdr_t*)cubin.sec_hdrs[i];
        if (sh->type == 3)
        {
            char *data = cubin.binary + sh->offset;
            char *name = data + sh->name;
            if (strcmp(name, ".strtab") == 0)        strtab = data;
            else if (strcmp(name, ".shstrtab") == 0) shstrtab = data;

            printf("found section \"%s\"\ndata (%d bytes): ", name, sh->size);
            for (int j = 0; j < sh->size; j++)
                printf("%c", data[j] ? data[j] : ' ');
            printf("\n\n");
        }
        #if 0
        else
        {
            char *name = shstrtab + sh->name;
            uint8_t *data = (uint8_t*)(cubin.binary + sh->offset);
            printf("found section \"%s\" (type=%x)\ndata(%d bytes):", name, sh->type, sh->size);
            for (int j = 0; j < sh->size; j++)
                printf("%02x ", data[j]);
            printf("\n\n");
        }
        #endif
    }
    assert(strtab);
    assert(shstrtab);

    for (int i = 0; i < cubin.num_sec_hdrs; i++)
    {
        elf64_sec_hdr_t *sh = cubin.sec_hdrs[i];
        if (sh->type == 2) // look for symbol table
        {
            printf("found symbol table section with these symbols:\n");
            char *data = cubin.binary + sh->offset;
            uint64_t offset = 0;
            while (offset < sh->size) // go through each symbol entry
            {
                elf64_sym_ent_t *ent = (elf64_sym_ent_t*)(data + offset);
                offset += sh->entSize;
                char *name = strtab + ent->name;

                if ((ent->info & 0x0f) == 0x02) // look for symbols tagged FUNC
                {
                    printf("(function) \"%s\"\n", name);
                    assert(cubin.num_functions < cubin_max_functions);
                    cubin_function_t func = {0};
                    func.name = name;
                    func.h    = cubin.sec_hdrs[ent->shIndx];
                    func.b    = cubin.binary;
                    func.e    = ent;
                    cubin.functions[cubin.num_functions++] = func;

                    // elf64_sec_hdr_t *ent_sh = cubin.sec_hdrs[ent->shIndx];
                    // printf("section header \"%s\"\n", strtab + ent_sh->name);
                }
                else
                {
                    printf("(other)    \"%s\"\n", name);
                }

                #if 0
                printf("\tinfo:0x%x\n", ent->info);
                printf("\tother:0x%x\n", ent->other);
                printf("\tvalue:0x%llx\n", ent->value);
                printf("\tsize:0x%llx (%llu)\n", ent->size, ent->size);
                #endif
            }
        }
    }

    printf("\nfound %d functions\n", cubin.num_functions);
    for (int i = 0; i < cubin.num_functions; i++)
    {
        printf("\"%s\"\n", cubin.functions[i].name);
        printf("\tRegister count: %d\n", cubin.functions[i].register_count());
        printf("\tInstructions:\n");
        uint64_t *in = cubin.functions[i].instructions();
        int num_instructions = cubin.functions[i].num_instructions();
        for (int j = 0; j < 10 && j < num_instructions; j++)
            printf("\t0x%016llx\n", in[j]);
        if (num_instructions > 10)
            printf("\t... (%d more instructions)\n", num_instructions - 10);
    }
    return cubin;
}

void save_cubin(cubin_t *cubin, const char *filename)
{
    FILE *f = fopen(filename, "wb+");
    assert(f);
    fwrite(cubin->binary, 1, cubin->sizeof_binary, f);
    fclose(f);
}
